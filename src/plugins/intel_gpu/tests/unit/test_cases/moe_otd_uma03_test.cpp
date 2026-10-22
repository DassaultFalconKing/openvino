// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// UMA-03: path A (usm_device) vs path B (usm_host) bandwidth / cold-miss timing on Arc.
// Scratch only — no production enablement, no win claim without operator review.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include "ocl_v2/moe/moe_otd_runtime.hpp"
#include "runtime/ocl/ocl_stream.hpp"

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
namespace moe_otd = ov::intel_gpu::ocl::moe_otd;

namespace {

class EnvVarGuard {
public:
    EnvVarGuard(const char* name, const char* value) : _name(name) {
#ifdef _WIN32
        char* prev = nullptr;
        size_t len = 0;
        if (_dupenv_s(&prev, &len, name) == 0 && prev != nullptr) {
            _had_prev = true;
            _prev.assign(prev);
            free(prev);
        }
        _putenv_s(name, value ? value : "");
#else
        const char* prev = std::getenv(name);
        if (prev) {
            _had_prev = true;
            _prev = prev;
        }
        if (value)
            setenv(name, value, 1);
        else
            unsetenv(name);
#endif
    }
    ~EnvVarGuard() {
#ifdef _WIN32
        if (_had_prev)
            _putenv_s(_name.c_str(), _prev.c_str());
        else
            _putenv_s(_name.c_str(), "");
#else
        if (_had_prev)
            setenv(_name.c_str(), _prev.c_str(), 1);
        else
            unsetenv(_name.c_str());
#endif
    }
    EnvVarGuard(const EnvVarGuard&) = delete;
    EnvVarGuard& operator=(const EnvVarGuard&) = delete;

private:
    std::string _name;
    bool _had_prev = false;
    std::string _prev;
};

// Reuse the same digest kernel contract as UMA-02 G01 (schedule-independent FNV windows).
constexpr uint32_t kDigestWorkItems = 64;
static const char kDigestKernelSource[] = R"(
__kernel void uma03_digest(__global const uchar* src, __global uint* dst, const uint begin, const uint n) {
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);
    const uint chunk = (n + gsz - 1u) / gsz;
    const uint start = begin + gid * chunk;
    const uint end = min(start + chunk, begin + n);
    uint acc = 0x811C9DC5u;
    for (uint i = start; i < end; i++) {
        acc ^= (uint)(src[i]);
        acc *= 0x01000193u;
    }
    dst[gid] = acc;
}
)";

cldnn::kernel::ptr get_uma03_digest_kernel() {
    static cldnn::kernel::ptr cached = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        auto builder = get_test_engine().create_kernel_builder();
        std::vector<cldnn::kernel::ptr> out;
        const char* src = kDigestKernelSource;
        builder->build_kernels(src, std::strlen(kDigestKernelSource), cldnn::KernelFormat::SOURCE, "", out);
        OPENVINO_ASSERT(!out.empty());
        cached = out.front();
    });
    return cached;
}

void enqueue_digest(const cldnn::memory::ptr& mem, uint32_t begin, uint32_t n, cldnn::memory::ptr& out_mem) {
    auto& stream = downcast<cldnn::ocl::ocl_stream>(get_test_stream());
    cldnn::kernel_arguments_desc desc;
    desc.workGroups.global = {kDigestWorkItems};
    desc.workGroups.local = {kDigestWorkItems};
    desc.arguments = {{cldnn::argument_desc::Types::INPUT, 0},
                      {cldnn::argument_desc::Types::OUTPUT, 0},
                      {cldnn::argument_desc::Types::SCALAR, 0},
                      {cldnn::argument_desc::Types::SCALAR, 1}};
    desc.layerID = "uma03_digest";
    cldnn::scalars_desc scalars;
    scalars.push_back({cldnn::scalar_desc::Types::UINT32, {0}});
    scalars.back().v.u32 = begin;
    scalars.push_back({cldnn::scalar_desc::Types::UINT32, {0}});
    scalars.back().v.u32 = n;
    desc.scalars = scalars;
    cldnn::kernel_arguments_data data;
    data.inputs = {mem};
    data.outputs = {out_mem};
    data.scalars = &scalars;
    auto kernel = get_uma03_digest_kernel();
    stream.set_arguments(*kernel, desc, data);
    (void)stream.enqueue_kernel(*kernel, desc, data, {});
}

uint64_t percentile_ns(std::vector<uint64_t> samples, double q) {
    OPENVINO_ASSERT(!samples.empty());
    std::sort(samples.begin(), samples.end());
    const size_t idx = static_cast<size_t>(std::min(samples.size() - 1, static_cast<size_t>(std::floor(q * (samples.size() - 1)))));
    return samples[idx];
}

class Uma03Fixture {
public:
    static constexpr size_t num_expert = 4;

    Uma03Fixture() {
        std::random_device rd;
        root = std::filesystem::temp_directory_path() / ("moe_otd_uma03_" + std::to_string(rd()) + std::to_string(rd()));
        pack_dir = root / "packs";
        std::filesystem::create_directories(pack_dir);
        weights_path = root / "weights.bin";

        config.hidden_size = 64;
        config.inter_size = 32;
        config.group_size = 32;
        config.num_expert = num_expert;
        config.top_k = 2;
        config.has_zp = true;

        const auto e = static_cast<int64_t>(num_expert);
        layouts = {layout{ov::PartialShape{e, 32, 2, 32}, data_types::u8, format::bfyx},
                   layout{ov::PartialShape{e, 32, 2, 32}, data_types::u8, format::bfyx},
                   layout{ov::PartialShape{e, 64, 1, 32}, data_types::u8, format::bfyx},
                   layout{ov::PartialShape{e, 32, 2, 1}, data_types::f16, format::bfyx},
                   layout{ov::PartialShape{e, 32, 2, 1}, data_types::f16, format::bfyx},
                   layout{ov::PartialShape{e, 64, 1, 1}, data_types::f16, format::bfyx},
                   layout{ov::PartialShape{e, 32, 2, 1}, data_types::u8, format::bfyx},
                   layout{ov::PartialShape{e, 32, 2, 1}, data_types::u8, format::bfyx},
                   layout{ov::PartialShape{e, 64, 1, 1}, data_types::u8, format::bfyx}};

        size_t cursor = 0;
        for (const auto& l : layouts) {
            offsets.push_back(cursor);
            cursor += l.bytes_count();
        }
        std::vector<char> bytes(cursor);
        for (size_t i = 0; i < bytes.size(); i++)
            bytes[i] = static_cast<char>((i * 31 + 7) & 0xFF);
        std::ofstream out(weights_path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    ~Uma03Fixture() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    std::unique_ptr<OffloadExpertWeightProvider> make_provider(size_t capacity = 2) const {
        return std::make_unique<OffloadExpertWeightProvider>(capacity, config, offsets, weights_path);
    }

    moe_otd::PackLayerSpec spec() const {
        moe_otd::ParallelWeightReader reader(weights_path);
        return moe_otd::make_pack_spec(config, offsets, layouts, reader);
    }

    void build_pack() const {
        auto provider = make_provider();
        provider->configure_pack(moe_otd::PackMode::build, pack_dir, layouts);
        OPENVINO_ASSERT(provider->uses_pack());
    }

    std::array<memory::ptr, moe_otd::pack_tensor_count> allocate_resident(cldnn::moe_weights& weights, allocation_type alloc) const {
        std::array<memory::ptr, moe_otd::pack_tensor_count> mems;
        for (size_t t = 0; t < mems.size(); t++) {
            mems[t] = get_test_engine().allocate_memory(layouts[t], alloc, false);
            cldnn::mem_lock<uint8_t, mem_lock_type::write> lock(mems[t], get_test_stream());
            std::fill(lock.begin(), lock.end(), 0x11);
        }
        weights.gate_w = mems[0];
        weights.up_w = mems[1];
        weights.down_w = mems[2];
        weights.gate_s = mems[3];
        weights.up_s = mems[4];
        weights.down_s = mems[5];
        weights.gate_z = mems[6];
        weights.up_z = mems[7];
        weights.down_z = mems[8];
        return mems;
    }

    std::filesystem::path root;
    std::filesystem::path pack_dir;
    std::filesystem::path weights_path;
    MOECompressed::Config config{};
    std::vector<size_t> offsets;
    moe_otd::RoutedTensorLayouts layouts;
};

}  // namespace

// U3-B01: steady-state GPU read bandwidth proxy — repeated digest over filled residents.
TEST(moe_otd_uma03, U3_B01_steady_kernel_read_usm_host_vs_device) {
    auto& engine = get_test_engine();
    std::cout << "[U3-B01] device=" << engine.get_device_info().dev_name << std::endl;

    Uma03Fixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_dir / fx.spec().file_name(), fx.spec());

    cldnn::layout out_layout({kDigestWorkItems}, data_types::i32, format::bfyx);
    auto out_mem = engine.allocate_memory(out_layout);

    auto time_path = [&](allocation_type alloc, const char* tag) {
        cldnn::moe_weights w;
        auto mems = fx.allocate_resident(w, alloc);
        auto provider = fx.make_provider();
        provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
        provider->bind(w);
        const size_t slot = provider->acquire_one(2, get_test_stream());
        get_test_stream().finish();

        const size_t bytes = static_cast<size_t>(pack.tensor_bytes(0));
        const uint32_t begin = static_cast<uint32_t>(slot * bytes);
        constexpr int warmup = 5;
        constexpr int iters = 40;
        for (int i = 0; i < warmup; i++) {
            enqueue_digest(mems[0], begin, static_cast<uint32_t>(bytes), out_mem);
            get_test_stream().finish();
        }
        std::vector<uint64_t> samples;
        samples.reserve(iters);
        for (int i = 0; i < iters; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            enqueue_digest(mems[0], begin, static_cast<uint32_t>(bytes), out_mem);
            get_test_stream().finish();
            samples.push_back(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
        }
        const uint64_t p50 = percentile_ns(samples, 0.50);
        const uint64_t p95 = percentile_ns(samples, 0.95);
        const double gbps = (bytes * 1.0) / (static_cast<double>(p50) * 1e-9) / 1e9;
        std::cout << "[U3-B01-" << tag << "] tensor0_bytes=" << bytes << " iters=" << iters << " p50_ns=" << p50
                  << " p95_ns=" << p95 << " approx_GB_s=" << gbps << " (digest proxy; not GEMM)" << std::endl;
        return p50;
    };

    const uint64_t device_p50 = time_path(allocation_type::usm_device, "PATH_A_DEVICE");
    const uint64_t host_p50 = time_path(allocation_type::usm_host, "PATH_B_HOST");
    const double ratio = static_cast<double>(host_p50) / static_cast<double>(std::max<uint64_t>(1, device_p50));
    std::cout << "[U3-B01] host_p50/device_p50=" << ratio << " ( >1 means host slower on this proxy )" << std::endl;
    std::cout << "[U3-B01-WARN] fixture-scale digest proxy only; not production MoE GEMM bandwidth" << std::endl;
    EXPECT_GT(device_p50, 0u);
    EXPECT_GT(host_p50, 0u);
}

// U3-B02: cold-miss wall + copy_ns path A vs B (extends G02 with repeats).
TEST(moe_otd_uma03, U3_B02_cold_miss_fill_timing_path_a_vs_b) {
    auto* perf = moe_otd::get_perf_counters();
    if (!perf) {
        GTEST_SKIP() << "MOE_OTD_PERF_LOG=1 required";
    }
    std::cout << "[U3-B02] device=" << get_test_engine().get_device_info().dev_name << std::endl;

    auto measure = [&](allocation_type alloc, const char* tag, bool path_b_env) {
        std::vector<uint64_t> cold_ns;
        std::vector<uint64_t> copy_ns;
        constexpr int iters = 12;
        for (int i = 0; i < iters; i++) {
            EnvVarGuard env("OTD_RESIDENT_USM_HOST", path_b_env ? "1" : "");
            Uma03Fixture fx;
            fx.build_pack();
            cldnn::moe_weights w;
            fx.allocate_resident(w, alloc);
            auto provider = fx.make_provider(/*capacity=*/1);
            provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
            provider->bind(w);

            // Rotate expert so each iteration is a cold miss into a recycled capacity-1 cache.
            const uint32_t expert = static_cast<uint32_t>(i % Uma03Fixture::num_expert);
            const auto before = perf->snapshot();
            const auto t0 = std::chrono::steady_clock::now();
            provider->acquire_one(expert, get_test_stream());
            get_test_stream().finish();
            const auto t1 = std::chrono::steady_clock::now();
            const auto after = perf->snapshot();
            cold_ns.push_back(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
            copy_ns.push_back(after.gpu_copy_ns - before.gpu_copy_ns);
        }
        const uint64_t cold_p50 = percentile_ns(cold_ns, 0.50);
        const uint64_t copy_p50 = percentile_ns(copy_ns, 0.50);
        std::cout << "[U3-B02-" << tag << "] cold_p50_ns=" << cold_p50 << " copy_p50_ns=" << copy_p50 << " n=" << iters
                  << std::endl;
        return std::make_pair(cold_p50, copy_p50);
    };

    const auto a = measure(allocation_type::usm_device, "PATH_A", false);
    const auto b = measure(allocation_type::usm_host, "PATH_B", true);
    std::cout << "[U3-B02] cold_ratio_B/A=" << (static_cast<double>(b.first) / std::max<uint64_t>(1, a.first))
              << " copy_ratio_B/A=" << (static_cast<double>(b.second) / std::max<uint64_t>(1, a.second)) << std::endl;
    std::cout << "[U3-B02-WARN] tiny pack / synthetic experts; do not treat as OVMS TTFT" << std::endl;
    EXPECT_GT(a.first, 0u);
    EXPECT_GT(b.first, 0u);
}
