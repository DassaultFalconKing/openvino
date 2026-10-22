// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Phase 4 — representative ornith-scale OTD cold-miss + quantized MoE GEMM/GEMV A/B.
// Path A: usm_device (flag off). Path B: usm_host via OTD_RESIDENT_USM_HOST=1.
// COMPUTE_GO: short only; flag default remains off in production code.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils/test_utils.h"
#include "random_generator.hpp"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include "ocl_v2/moe/moe_otd_runtime.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/moe_gemm.hpp>
#include "ov_ops/moe_compressed.hpp"

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

uint64_t percentile_ns(std::vector<uint64_t> samples, double q) {
    OPENVINO_ASSERT(!samples.empty());
    std::sort(samples.begin(), samples.end());
    const size_t idx =
        static_cast<size_t>(std::min(samples.size() - 1, static_cast<size_t>(std::floor(q * (samples.size() - 1)))));
    return samples[idx];
}

// Ornith-35b-ov expert record geometry (from live pack header): record_stride ≈ 1.617 MiB.
class OrnithScaleFixture {
public:
    static constexpr size_t num_expert = 4;
    static constexpr size_t hidden_size = 2048;
    static constexpr size_t inter_size = 512;
    static constexpr size_t group_size = 64;

    OrnithScaleFixture() {
        std::random_device rd;
        root = std::filesystem::temp_directory_path() / ("moe_otd_gemm_ab_" + std::to_string(rd()) + std::to_string(rd()));
        pack_dir = root / "packs";
        std::filesystem::create_directories(pack_dir);
        weights_path = root / "weights.bin";

        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.group_size = group_size;
        config.num_expert = num_expert;
        config.top_k = 2;
        config.has_zp = true;

        const auto e = static_cast<int64_t>(num_expert);
        const auto g_h = static_cast<int64_t>(hidden_size / group_size);  // 32
        const auto g_i = static_cast<int64_t>(inter_size / group_size);   // 8
        // u4 packed as half-bytes → u8 layout with K/2 in the group dim (matches 524288 B / expert).
        layouts = {layout{ov::PartialShape{e, static_cast<int64_t>(inter_size), g_h, static_cast<int64_t>(group_size / 2)},
                          data_types::u8,
                          format::bfyx},
                   layout{ov::PartialShape{e, static_cast<int64_t>(inter_size), g_h, static_cast<int64_t>(group_size / 2)},
                          data_types::u8,
                          format::bfyx},
                   layout{ov::PartialShape{e, static_cast<int64_t>(hidden_size), g_i, static_cast<int64_t>(group_size / 2)},
                          data_types::u8,
                          format::bfyx},
                   layout{ov::PartialShape{e, static_cast<int64_t>(inter_size), g_h, 1}, data_types::f16, format::bfyx},
                   layout{ov::PartialShape{e, static_cast<int64_t>(inter_size), g_h, 1}, data_types::f16, format::bfyx},
                   layout{ov::PartialShape{e, static_cast<int64_t>(hidden_size), g_i, 1}, data_types::f16, format::bfyx},
                   // ZP matches ornith pack: u4 → 8192 B/expert (gate/up) and 8192 B (down).
                   layout{ov::PartialShape{e, static_cast<int64_t>(inter_size), g_h, 1}, data_types::u4, format::bfyx},
                   layout{ov::PartialShape{e, static_cast<int64_t>(inter_size), g_h, 1}, data_types::u4, format::bfyx},
                   layout{ov::PartialShape{e, static_cast<int64_t>(hidden_size), g_i, 1}, data_types::u4, format::bfyx}};

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

    ~OrnithScaleFixture() {
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

    std::array<memory::ptr, moe_otd::pack_tensor_count> allocate_resident(cldnn::moe_weights& weights,
                                                                           allocation_type alloc) const {
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

struct PathTiming {
    uint64_t cold_p50_ns = 0;
    uint64_t cold_p95_ns = 0;
    uint64_t copy_p50_ns = 0;
    uint64_t pre_write_calls = 0;
    uint64_t fence_calls = 0;
    uint64_t record_bytes = 0;
};

PathTiming measure_cold_miss(allocation_type alloc, bool path_b_env, const char* tag) {
    auto* perf = moe_otd::get_perf_counters();
    OPENVINO_ASSERT(perf != nullptr, "MOE_OTD_PERF_LOG=1 required");

    moe_otd::reset_path_b_harness_counters();
    std::vector<uint64_t> cold_ns;
    std::vector<uint64_t> copy_ns;
    uint64_t record_bytes = 0;
    constexpr int iters = 8;

    for (int i = 0; i < iters; i++) {
        EnvVarGuard env("OTD_RESIDENT_USM_HOST", path_b_env ? "1" : "");
        OrnithScaleFixture fx;
        fx.build_pack();
        if (record_bytes == 0) {
            const auto pack = *moe_otd::PackLayout::open(fx.pack_dir / fx.spec().file_name(), fx.spec());
            record_bytes = pack.record_stride();
        }
        cldnn::moe_weights w;
        fx.allocate_resident(w, alloc);
        auto provider = fx.make_provider(/*capacity=*/1);
        provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
        provider->bind(w);

        const uint32_t expert = static_cast<uint32_t>(i % OrnithScaleFixture::num_expert);
        const auto before = perf->snapshot();
        const auto pre0 = moe_otd::path_b_pre_write_calls();
        const auto fence0 = moe_otd::path_b_fence_calls();
        const auto t0 = std::chrono::steady_clock::now();
        provider->acquire_one(expert, get_test_stream());
        get_test_stream().finish();
        const auto t1 = std::chrono::steady_clock::now();
        const auto after = perf->snapshot();
        cold_ns.push_back(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
        copy_ns.push_back(after.gpu_copy_ns - before.gpu_copy_ns);
        (void)pre0;
        (void)fence0;
    }

    PathTiming out;
    out.cold_p50_ns = percentile_ns(cold_ns, 0.50);
    out.cold_p95_ns = percentile_ns(cold_ns, 0.95);
    out.copy_p50_ns = percentile_ns(copy_ns, 0.50);
    out.pre_write_calls = moe_otd::path_b_pre_write_calls();
    out.fence_calls = moe_otd::path_b_fence_calls();
    out.record_bytes = record_bytes;
    std::cout << "[U4-COLD-" << tag << "] record_bytes=" << out.record_bytes << " (~"
              << (out.record_bytes / (1024.0 * 1024.0)) << " MiB) cold_p50_ns=" << out.cold_p50_ns
              << " cold_p95_ns=" << out.cold_p95_ns << " copy_p50_ns=" << out.copy_p50_ns
              << " path_b_pre_write_calls=" << out.pre_write_calls << " path_b_fence_calls=" << out.fence_calls
              << std::endl;
    return out;
}

struct GemmTiming {
    size_t M = 0;
    uint64_t p50_ns = 0;
    uint64_t p95_ns = 0;
};

// Quantized MoE GEMM timing with explicit USM alloc for expert weights (path A/B proxy).
std::vector<GemmTiming> measure_quant_gemm(allocation_type weight_alloc, const char* tag) {
    auto& engine = get_test_engine();
    std::vector<GemmTiming> results;
    if (!engine.get_device_info().supports_immad) {
        std::cout << "[U4-GEMM-" << tag << "] SKIP supports_immad=false" << std::endl;
        return results;
    }

    constexpr size_t E = 4;
    constexpr size_t hidden = 2048;
    constexpr size_t out_N = 512;
    constexpr size_t group = 64;
    constexpr size_t groups = hidden / group;
    const std::array<size_t, 5> Ms = {1, 2, 4, 8, 11};

    // Build once: i4 symmetric weights + scales (smoke shape used by moe_gemm_gpu_test).
    ov::op::internal::MOECompressed::Config moe_config;
    moe_config.top_k = 1;
    moe_config.num_expert = E;
    moe_config.has_batch_dim = true;
    moe_config.hidden_size = hidden;
    moe_config.inter_size = out_N;

    auto experts_shape =
        ov::PartialShape{ov::Dimension(E), ov::Dimension(out_N), ov::Dimension(groups), ov::Dimension(group)};
    auto experts_layout = layout{experts_shape, data_types::i4, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout, weight_alloc, false);
    {
        cldnn::mem_lock<uint8_t, mem_lock_type::write> lock(experts_mem, get_test_stream());
        for (size_t i = 0; i < lock.size(); i++)
            lock[i] = static_cast<uint8_t>((i * 17 + 3) & 0xFF);
    }

    auto scale_shape = ov::PartialShape{ov::Dimension(E), ov::Dimension(out_N), ov::Dimension(groups), ov::Dimension(1)};
    auto scale_layout = layout{scale_shape, data_types::f16, format::byfx};
    auto scale_mem = engine.allocate_memory(scale_layout, weight_alloc, false);
    {
        cldnn::mem_lock<ov::float16, mem_lock_type::write> lock(scale_mem, get_test_stream());
        for (size_t i = 0; i < lock.size(); i++)
            lock[i] = ov::float16(0.01f);
    }

    std::mt19937 rng(0x4AB);
    std::vector<size_t> order(Ms.begin(), Ms.end());
    std::shuffle(order.begin(), order.end(), rng);

    for (size_t M : order) {
        topology topo;
        auto input_shape = ov::PartialShape{ov::Dimension(1), ov::Dimension(static_cast<int64_t>(M)), ov::Dimension(hidden)};
        topo.add(input_layout("input", layout{input_shape, data_types::f16, format::bfyx}));
        topo.add(input_layout("experts_ids", layout{ov::PartialShape{ov::Dimension::dynamic()}, data_types::i32, format::bfyx}));
        topo.add(input_layout("input_offset_per_expert",
                              layout{ov::PartialShape{ov::Dimension::dynamic()}, data_types::i32, format::bfyx}));
        topo.add(input_layout("input_tokens_lens",
                              layout{ov::PartialShape{ov::Dimension::dynamic()}, data_types::i32, format::bfyx}));
        topo.add(data("moe_experts", experts_mem));
        topo.add(data("moe_experts_scale", scale_mem));
        std::vector<input_info> inputs = {input_info("input"),
                                          input_info("moe_experts"),
                                          input_info("experts_ids"),
                                          input_info("input_offset_per_expert"),
                                          input_info("input_tokens_lens"),
                                          input_info("moe_experts_scale")};
        auto gemm = moe_gemm("moe_gemm", inputs, moe_config);
        topo.add(gemm);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

        auto input_mem =
            engine.allocate_memory(layout{ov::PartialShape{1, static_cast<int64_t>(M), static_cast<int64_t>(hidden)},
                                          data_types::f16,
                                          format::bfyx});
        {
            cldnn::mem_lock<ov::float16, mem_lock_type::write> lock(input_mem, get_test_stream());
            for (size_t i = 0; i < lock.size(); i++)
                lock[i] = ov::float16(static_cast<float>((i % 20) + 1) / 10.f);
        }
        auto experts_ids_mem = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::i32, format::bfyx});
        set_values(experts_ids_mem, std::vector<int32_t>{0});
        auto tokens_lens_mem = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::i32, format::bfyx});
        set_values(tokens_lens_mem, std::vector<int32_t>{static_cast<int32_t>(M)});
        std::vector<int32_t> offsets(E, 0);
        offsets[0] = static_cast<int32_t>(M);
        for (size_t e = 1; e < E; e++)
            offsets[e] = offsets[e - 1];
        auto offsets_mem =
            engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(E)}, data_types::i32, format::bfyx});
        set_values(offsets_mem, offsets);

        network->set_input_data("input", input_mem);
        network->set_input_data("experts_ids", experts_ids_mem);
        network->set_input_data("input_offset_per_expert", offsets_mem);
        network->set_input_data("input_tokens_lens", tokens_lens_mem);

        constexpr int warmup = 3;
        constexpr int iters = 12;
        for (int i = 0; i < warmup; i++) {
            (void)network->execute();
            get_test_stream().finish();
        }
        std::vector<uint64_t> samples;
        samples.reserve(iters);
        for (int i = 0; i < iters; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            (void)network->execute();
            get_test_stream().finish();
            samples.push_back(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
        }
        GemmTiming gt;
        gt.M = M;
        gt.p50_ns = percentile_ns(samples, 0.50);
        gt.p95_ns = percentile_ns(samples, 0.95);
        std::cout << "[U4-GEMM-" << tag << "] M=" << M << " weight_alloc="
                  << (weight_alloc == allocation_type::usm_host ? "usm_host" : "usm_device")
                  << " p50_ns=" << gt.p50_ns << " p95_ns=" << gt.p95_ns << std::endl;
        results.push_back(gt);
    }
    return results;
}

}  // namespace

TEST(moe_otd_gemm_ab, U4_COLD_ornith_record_path_a_vs_b) {
    if (!moe_otd::get_perf_counters()) {
        GTEST_SKIP() << "MOE_OTD_PERF_LOG=1 required";
    }
    std::cout << "[U4-COLD] device=" << get_test_engine().get_device_info().dev_name << std::endl;

    // Clear flag for path A measurement.
    EnvVarGuard clear("OTD_RESIDENT_USM_HOST", "");
    const auto a = measure_cold_miss(allocation_type::usm_device, false, "PATH_A");
    const auto b = measure_cold_miss(allocation_type::usm_host, true, "PATH_B");

    EXPECT_GE(a.record_bytes, 512ull * 1024ull);
    EXPECT_LE(a.record_bytes, 4ull * 1024ull * 1024ull);
    EXPECT_GT(a.cold_p50_ns, 0u);
    EXPECT_GT(b.cold_p50_ns, 0u);
    EXPECT_GE(b.pre_write_calls, 1u);
    EXPECT_GE(b.fence_calls, 1u);

    const double cold_ratio = static_cast<double>(b.cold_p50_ns) / static_cast<double>(std::max<uint64_t>(1, a.cold_p50_ns));
    const double copy_ratio = static_cast<double>(b.copy_p50_ns) / static_cast<double>(std::max<uint64_t>(1, a.copy_p50_ns));
    std::cout << "[U4-COLD] cold_ratio_B/A=" << cold_ratio << " copy_ratio_B/A=" << copy_ratio << std::endl;
}

TEST(moe_otd_gemm_ab, U4_GEMM_M_sweep_path_a_vs_b) {
    std::cout << "[U4-GEMM] device=" << get_test_engine().get_device_info().dev_name << std::endl;
    // Default OFF — path A does not enable the env flag; weight USM type is the bandwidth proxy.
    EnvVarGuard clear("OTD_RESIDENT_USM_HOST", "");
    ASSERT_FALSE(moe_otd::otd_resident_usm_host_enabled());

    const auto a = measure_quant_gemm(allocation_type::usm_device, "PATH_A");
    {
        EnvVarGuard on("OTD_RESIDENT_USM_HOST", "1");
        ASSERT_TRUE(moe_otd::otd_resident_usm_host_enabled());
        const auto b = measure_quant_gemm(allocation_type::usm_host, "PATH_B");

        if (a.empty() || b.empty()) {
            GTEST_SKIP() << "moe_gemm unavailable (supports_immad=false)";
        }
        ASSERT_EQ(a.size(), b.size());

        double worst_ratio = 0.0;
        double sum_a = 0.0;
        double sum_b = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            // Match by M (randomized order).
            const auto* ba = &a[i];
            const GemmTiming* bb = nullptr;
            for (const auto& x : b) {
                if (x.M == ba->M) {
                    bb = &x;
                    break;
                }
            }
            ASSERT_NE(bb, nullptr);
            const double ratio = static_cast<double>(bb->p50_ns) / static_cast<double>(std::max<uint64_t>(1, ba->p50_ns));
            worst_ratio = std::max(worst_ratio, ratio);
            sum_a += static_cast<double>(ba->p50_ns);
            sum_b += static_cast<double>(bb->p50_ns);
            std::cout << "[U4-GEMM-RATIO] M=" << ba->M << " B/A_p50=" << ratio << std::endl;
        }
        const double mean_ratio = sum_b / std::max(1.0, sum_a);
        std::cout << "[U4-GEMM-SUMMARY] mean_B/A_p50=" << mean_ratio << " worst_B/A_p50=" << worst_ratio << std::endl;
        // Gate hint for operator: steady B loses enough to erase copy savings → B_PERF_REJECTED.
        // Threshold: mean steady GEMM B/A > 1.25 (25% slower) is a strong reject signal when
        // combined with cold-miss savings; printed for verdict doc, not a hard ASSERT.
        if (mean_ratio > 1.25) {
            std::cout << "[U4-GEMM-VERDICT-HINT] B_PERF_REJECTED_CANDIDATE mean_ratio=" << mean_ratio << std::endl;
        } else {
            std::cout << "[U4-GEMM-VERDICT-HINT] B_STEADY_ACCEPTABLE mean_ratio=" << mean_ratio << std::endl;
        }
        EXPECT_GT(sum_a, 0.0);
        EXPECT_GT(sum_b, 0.0);
    }
    // Flag must not leak.
    EXPECT_FALSE(moe_otd::otd_resident_usm_host_enabled());
}
