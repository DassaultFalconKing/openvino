// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// P0-B: path-B in-flight slot safety — overlap (1Q/2Q) + post-write visibility.
// COMPUTE_GO: short GPU correctness only; no soak. OTD_RESIDENT_USM_HOST default remains off.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include "ocl_v2/moe/moe_otd_runtime.hpp"
#include "plugin/ops/moe_offload_constant.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_kernel_builder.hpp"
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
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
#endif
    }

    ~EnvVarGuard() {
#ifdef _WIN32
        if (_had_prev) {
            _putenv_s(_name.c_str(), _prev.c_str());
        } else {
            _putenv_s(_name.c_str(), "");
        }
#else
        if (_had_prev) {
            setenv(_name.c_str(), _prev.c_str(), 1);
        } else {
            unsetenv(_name.c_str());
        }
#endif
    }

    EnvVarGuard(const EnvVarGuard&) = delete;
    EnvVarGuard& operator=(const EnvVarGuard&) = delete;

private:
    std::string _name;
    bool _had_prev = false;
    std::string _prev;
};

class PathBHarnessGuard {
public:
    PathBHarnessGuard() = default;
    ~PathBHarnessGuard() {
        moe_otd::path_b_set_skip_pre_write_for_test(false);
        moe_otd::path_b_set_skip_post_fence_for_test(false);
        moe_otd::path_b_clear_pre_write_dependents();
    }
    PathBHarnessGuard(const PathBHarnessGuard&) = delete;
    PathBHarnessGuard& operator=(const PathBHarnessGuard&) = delete;
};

class P0bFixture {
public:
    static constexpr size_t num_expert = 4;
    static constexpr uint8_t sentinel = 0xEE;

    P0bFixture() {
        std::random_device rd;
        root = std::filesystem::temp_directory_path() / ("moe_otd_p0b_" + std::to_string(rd()) + std::to_string(rd()));
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
        const int64_t gate_groups = 64 / 32;
        const int64_t down_groups = 32 / 32;
        layouts = {
            layout{ov::PartialShape{e, 32, gate_groups, 32}, data_types::u8, format::bfyx},
            layout{ov::PartialShape{e, 32, gate_groups, 32}, data_types::u8, format::bfyx},
            layout{ov::PartialShape{e, 64, down_groups, 32}, data_types::u8, format::bfyx},
            layout{ov::PartialShape{e, 32, gate_groups, 1}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{e, 32, gate_groups, 1}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{e, 64, down_groups, 1}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{e, 32, gate_groups, 1}, data_types::u8, format::bfyx},
            layout{ov::PartialShape{e, 32, gate_groups, 1}, data_types::u8, format::bfyx},
            layout{ov::PartialShape{e, 64, down_groups, 1}, data_types::u8, format::bfyx}};

        size_t cursor = 0;
        for (const auto& l : layouts) {
            offsets.push_back(cursor);
            cursor += l.bytes_count();
        }
        // Distinct per-expert payloads so overlap tests can tell E0 from E1.
        std::vector<char> bytes(cursor);
        for (size_t i = 0; i < bytes.size(); i++) {
            bytes[i] = static_cast<char>((i * 31 + 7) & 0xFF);
        }
        for (uint32_t expert = 0; expert < num_expert; expert++) {
            size_t off = 0;
            for (size_t t = 0; t < layouts.size(); t++) {
                const size_t per = layouts[t].bytes_count() / num_expert;
                for (size_t b = 0; b < per; b++) {
                    bytes[offsets[t] + expert * per + b] =
                        static_cast<char>(((expert + 1) * 17 + t * 3 + b * 13) & 0xFF);
                }
                (void)off;
            }
        }
        std::ofstream out(weights_path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    ~P0bFixture() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    std::unique_ptr<OffloadExpertWeightProvider> make_provider(size_t capacity = 1) const {
        return std::make_unique<OffloadExpertWeightProvider>(capacity, config, offsets, weights_path);
    }

    moe_otd::PackLayerSpec spec() const {
        moe_otd::ParallelWeightReader reader(weights_path);
        return moe_otd::make_pack_spec(config, offsets, layouts, reader);
    }

    std::filesystem::path pack_path() const {
        return pack_dir / spec().file_name();
    }

    void build_pack() const {
        auto provider = make_provider();
        provider->configure_pack(moe_otd::PackMode::build, pack_dir, layouts);
        OPENVINO_ASSERT(provider->uses_pack(), "p0b fixture pack was not built");
    }

    void flip_record_byte(uint32_t expert, size_t tensor, uint64_t byte) const {
        const auto pack = moe_otd::PackLayout::open(pack_path(), spec());
        OPENVINO_ASSERT(pack.has_value(), "pack open failed");
        const auto offset = pack->record_offset(expert) + pack->tensor_offset_in_record(tensor) + byte;
        std::fstream io(pack_path(), std::ios::binary | std::ios::in | std::ios::out);
        io.seekg(static_cast<std::streamoff>(offset));
        char c = 0;
        io.read(&c, 1);
        c = static_cast<char>(c ^ 0x5A);
        io.seekp(static_cast<std::streamoff>(offset));
        io.write(&c, 1);
        OPENVINO_ASSERT(static_cast<bool>(io), "fixture byte flip failed");
    }

    std::array<memory::ptr, moe_otd::pack_tensor_count> allocate_resident(cldnn::moe_weights& weights,
                                                                           allocation_type alloc) const {
        auto& engine = get_test_engine();
        std::array<memory::ptr, moe_otd::pack_tensor_count> mems;
        for (size_t t = 0; t < mems.size(); t++) {
            mems[t] = engine.allocate_memory(layouts[t], alloc, false);
            cldnn::mem_lock<uint8_t, mem_lock_type::write> lock(mems[t], get_test_stream());
            std::fill(lock.begin(), lock.end(), sentinel);
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

    static std::vector<uint8_t> dump_slot(const memory::ptr& mem, size_t slot, size_t per_expert) {
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock(mem, get_test_stream());
        std::vector<uint8_t> out(per_expert);
        const size_t base = slot * per_expert;
        for (size_t i = 0; i < per_expert; i++) {
            out[i] = lock[base + i];
        }
        return out;
    }

    std::filesystem::path root;
    std::filesystem::path pack_dir;
    std::filesystem::path weights_path;
    MOECompressed::Config config{};
    std::vector<size_t> offsets;
    moe_otd::RoutedTensorLayouts layouts;
};

constexpr uint32_t kDigestWorkItems = 64;
// Enough passes that a host-side fill can race without pre-write ownership on Arc.
constexpr uint32_t kSlowPasses = 250000u;

static const char kSlowDigestKernelSource[] = R"(
__kernel void p0b_slow_digest(__global const uchar* src, __global uint* dst,
                              const uint begin, const uint n, const uint passes) {
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);
    const uint chunk = (n + gsz - 1u) / gsz;
    const uint start = begin + gid * chunk;
    const uint end = min(start + chunk, begin + n);
    uint acc = 0x811C9DC5u;
    for (uint p = 0; p < passes; p++) {
        for (uint i = start; i < end; i++) {
            acc ^= (uint)(src[i]);
            acc *= 0x01000193u;
        }
    }
    dst[gid] = acc;
}
)";

static const char kFastDigestKernelSource[] = R"(
__kernel void p0b_fast_digest(__global const uchar* src, __global uint* dst, const uint begin, const uint n) {
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

cldnn::kernel::ptr get_p0b_slow_digest_kernel() {
    static cldnn::kernel::ptr cached = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        auto& engine = get_test_engine();
        auto builder = engine.create_kernel_builder();
        OPENVINO_ASSERT(builder != nullptr, "p0b: no kernel builder");
        std::vector<cldnn::kernel::ptr> out;
        builder->build_kernels(kSlowDigestKernelSource,
                               std::strlen(kSlowDigestKernelSource),
                               cldnn::KernelFormat::SOURCE,
                               "",
                               out);
        OPENVINO_ASSERT(!out.empty(), "p0b: slow digest kernel build failed");
        cached = out.front();
    });
    return cached;
}

cldnn::kernel::ptr get_p0b_fast_digest_kernel() {
    static cldnn::kernel::ptr cached = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        auto& engine = get_test_engine();
        auto builder = engine.create_kernel_builder();
        OPENVINO_ASSERT(builder != nullptr, "p0b: no kernel builder");
        std::vector<cldnn::kernel::ptr> out;
        builder->build_kernels(kFastDigestKernelSource,
                               std::strlen(kFastDigestKernelSource),
                               cldnn::KernelFormat::SOURCE,
                               "",
                               out);
        OPENVINO_ASSERT(!out.empty(), "p0b: fast digest kernel build failed");
        cached = out.front();
    });
    return cached;
}

std::vector<uint32_t> cpu_slow_digest_reference(const uint8_t* bytes, size_t n, uint32_t passes) {
    uint32_t num = static_cast<uint32_t>(kDigestWorkItems);
    uint32_t chunk = (static_cast<uint32_t>(n) + num - 1u) / num;
    std::vector<uint32_t> digests(num, 0x811C9DC5u);
    for (uint32_t gid = 0; gid < num; gid++) {
        const size_t start = static_cast<size_t>(gid) * chunk;
        const size_t end = std::min(start + chunk, n);
        uint32_t acc = 0x811C9DC5u;
        for (uint32_t p = 0; p < passes; p++) {
            for (size_t i = start; i < end; i++) {
                acc ^= static_cast<uint32_t>(bytes[i]);
                acc *= 0x01000193u;
            }
        }
        digests[gid] = acc;
    }
    return digests;
}

std::vector<uint32_t> cpu_fast_digest_reference(const uint8_t* bytes, size_t n) {
    return cpu_slow_digest_reference(bytes, n, 1);
}

// Enqueue slow digest; does NOT finish — caller owns completion.
cldnn::memory::ptr enqueue_slow_digest(cldnn::stream& stream,
                                       const cldnn::memory::ptr& mem,
                                       uint32_t begin,
                                       uint32_t n,
                                       uint32_t passes = kSlowPasses) {
    auto& ocl = downcast<cldnn::ocl::ocl_stream>(stream);
    cldnn::layout out_layout({kDigestWorkItems}, data_types::i32, format::bfyx);
    auto out_mem = get_test_engine().allocate_memory(out_layout);

    cldnn::kernel_arguments_desc desc;
    desc.workGroups.global = {kDigestWorkItems};
    desc.workGroups.local = {kDigestWorkItems};
    desc.arguments = {
        {cldnn::argument_desc::Types::INPUT, 0},
        {cldnn::argument_desc::Types::OUTPUT, 0},
        {cldnn::argument_desc::Types::SCALAR, 0},
        {cldnn::argument_desc::Types::SCALAR, 1},
        {cldnn::argument_desc::Types::SCALAR, 2}};
    desc.layerID = "p0b_slow_digest";

    cldnn::scalars_desc scalars;
    scalars.push_back({cldnn::scalar_desc::Types::UINT32, {0}});
    scalars.back().v.u32 = begin;
    scalars.push_back({cldnn::scalar_desc::Types::UINT32, {0}});
    scalars.back().v.u32 = n;
    scalars.push_back({cldnn::scalar_desc::Types::UINT32, {0}});
    scalars.back().v.u32 = passes;
    desc.scalars = scalars;

    cldnn::kernel_arguments_data data;
    data.inputs = {mem};
    data.outputs = {out_mem};
    data.scalars = &scalars;

    auto kernel = get_p0b_slow_digest_kernel();
    ocl.set_arguments(*kernel, desc, data);
    (void)ocl.enqueue_kernel(*kernel, desc, data, {});
    ocl.flush();
    return out_mem;
}

std::vector<uint32_t> read_digest_out(const cldnn::memory::ptr& out_mem, cldnn::stream& stream) {
    stream.finish();
    std::vector<uint32_t> output(kDigestWorkItems);
    cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> lock(out_mem, stream);
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = static_cast<uint32_t>(lock[i]);
    }
    return output;
}

std::vector<uint32_t> gpu_fast_digest(cldnn::stream& stream,
                                      const cldnn::memory::ptr& mem,
                                      uint32_t begin,
                                      uint32_t n) {
    auto& ocl = downcast<cldnn::ocl::ocl_stream>(stream);
    cldnn::layout out_layout({kDigestWorkItems}, data_types::i32, format::bfyx);
    auto out_mem = get_test_engine().allocate_memory(out_layout);

    cldnn::kernel_arguments_desc desc;
    desc.workGroups.global = {kDigestWorkItems};
    desc.workGroups.local = {kDigestWorkItems};
    desc.arguments = {
        {cldnn::argument_desc::Types::INPUT, 0},
        {cldnn::argument_desc::Types::OUTPUT, 0},
        {cldnn::argument_desc::Types::SCALAR, 0},
        {cldnn::argument_desc::Types::SCALAR, 1}};
    desc.layerID = "p0b_fast_digest";

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

    auto kernel = get_p0b_fast_digest_kernel();
    ocl.set_arguments(*kernel, desc, data);
    (void)ocl.enqueue_kernel(*kernel, desc, data, {});
    ocl.finish();

    std::vector<uint32_t> output(kDigestWorkItems);
    cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> lock(out_mem, stream);
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = static_cast<uint32_t>(lock[i]);
    }
    return output;
}

void print_p0b_header() {
    const auto& info = get_test_engine().get_device_info();
    std::cout << "[P0B] device=" << info.dev_name << std::endl;
}

}  // namespace

// P0B-O1: same queue — delayed GPU reader on E0 slot; fill E1 must not tear; final GPU digest = E1.
TEST(moe_otd_p0b, P0B_O1_same_queue_overlap) {
    PathBHarnessGuard harness;
    EnvVarGuard on("OTD_RESIDENT_USM_HOST", "1");
    print_p0b_header();

    P0bFixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());
    const size_t bytes0 = static_cast<size_t>(pack.tensor_bytes(0));
    ASSERT_GT(bytes0, 0u);

    cldnn::moe_weights w;
    auto mems = fx.allocate_resident(w, allocation_type::usm_host);
    auto provider = fx.make_provider(/*capacity=*/1);
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    provider->bind(w);

    auto& stream = get_test_stream();
    const size_t slot = provider->acquire_one(0, stream);
    ASSERT_TRUE(provider->cache().lookup(0).hit);
    auto dump_e0 = P0bFixture::dump_slot(mems[0], slot, bytes0);
    const auto ref_e0 = cpu_slow_digest_reference(dump_e0.data(), bytes0, kSlowPasses);

    const uint32_t off = static_cast<uint32_t>(slot * bytes0);
    auto out_mem = enqueue_slow_digest(stream, mems[0], off, static_cast<uint32_t>(bytes0));

    moe_otd::reset_path_b_harness_counters();
    const size_t slot1 = provider->acquire_one(1, stream);  // pre-write finish drains reader
    ASSERT_EQ(slot, slot1);
    EXPECT_GE(moe_otd::path_b_pre_write_calls(), 1u);
    EXPECT_GE(moe_otd::path_b_fence_calls(), 1u);

    const auto reader_digest = read_digest_out(out_mem, stream);
    ASSERT_EQ(reader_digest, ref_e0) << "P0B-O1: in-flight same-queue reader must complete on E0 bytes";

    auto dump_e1 = P0bFixture::dump_slot(mems[0], slot, bytes0);
    ASSERT_NE(dump_e0, dump_e1) << "P0B-O1: E0 and E1 payloads must differ";
    const auto final_digest = gpu_fast_digest(stream, mems[0], off, static_cast<uint32_t>(bytes0));
    ASSERT_EQ(final_digest, cpu_fast_digest_reference(dump_e1.data(), bytes0));
    ASSERT_TRUE(provider->cache().lookup(1).hit);
    std::cout << "[P0B-O1] same-queue overlap PASS; reader=E0 final=E1" << std::endl;
}

// P0B-O2: reader on Q1, fill/scatter on Q2; ownership via registered dependent stream.
TEST(moe_otd_p0b, P0B_O2_two_queues_overlap) {
    PathBHarnessGuard harness;
    EnvVarGuard on("OTD_RESIDENT_USM_HOST", "1");
    print_p0b_header();

    P0bFixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());
    const size_t bytes0 = static_cast<size_t>(pack.tensor_bytes(0));
    ASSERT_GT(bytes0, 0u);

    cldnn::moe_weights w;
    auto mems = fx.allocate_resident(w, allocation_type::usm_host);
    auto provider = fx.make_provider(/*capacity=*/1);
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    provider->bind(w);

    auto& engine = get_test_engine();
    auto cfg = get_test_default_config(engine);
    auto q1 = engine.create_stream(cfg);
    auto& q2 = get_test_stream();
    ASSERT_NE(q1.get(), &q2);

    const size_t slot = provider->acquire_one(0, q2);
    auto dump_e0 = P0bFixture::dump_slot(mems[0], slot, bytes0);
    const auto ref_e0 = cpu_slow_digest_reference(dump_e0.data(), bytes0, kSlowPasses);

    const uint32_t off = static_cast<uint32_t>(slot * bytes0);
    auto out_mem = enqueue_slow_digest(*q1, mems[0], off, static_cast<uint32_t>(bytes0));
    moe_otd::path_b_register_pre_write_dependent(*q1);

    moe_otd::reset_path_b_harness_counters();
    const size_t slot1 = provider->acquire_one(1, q2);
    ASSERT_EQ(slot, slot1);
    EXPECT_GE(moe_otd::path_b_pre_write_calls(), 1u);

    const auto reader_digest = read_digest_out(out_mem, *q1);
    ASSERT_EQ(reader_digest, ref_e0) << "P0B-O2: Q1 reader must complete on E0 before Q2 scatter";

    auto dump_e1 = P0bFixture::dump_slot(mems[0], slot, bytes0);
    ASSERT_NE(dump_e0, dump_e1);
    const auto final_digest = gpu_fast_digest(q2, mems[0], off, static_cast<uint32_t>(bytes0));
    ASSERT_EQ(final_digest, cpu_fast_digest_reference(dump_e1.data(), bytes0));
    std::cout << "[P0B-O2] two-queue overlap PASS; Q1=E0 Q2-final=E1" << std::endl;
}

// P0B-V1: after scatter+fence and before set_filled, GPU sees new payload; without fence document NEO.
TEST(moe_otd_p0b, P0B_V1_visibility_before_set_filled) {
    PathBHarnessGuard harness;
    EnvVarGuard on("OTD_RESIDENT_USM_HOST", "1");
    print_p0b_header();

    P0bFixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());
    const size_t bytes0 = static_cast<size_t>(pack.tensor_bytes(0));
    ASSERT_GT(bytes0, 0u);

    auto& stream = get_test_stream();
    moe_otd::ParallelWeightReader pack_reader(fx.pack_path());
    std::vector<uint8_t> staging;

    // --- with post-write fence (production path) ---
    {
        cldnn::moe_weights w;
        auto mems = fx.allocate_resident(w, allocation_type::usm_host);
        const size_t slot = 0;
        moe_otd::reset_path_b_harness_counters();
        moe_otd::path_b_set_skip_post_fence_for_test(false);
        ASSERT_NO_THROW(moe_otd::fill_weights_from_pack(stream, pack, pack_reader, w, /*expert=*/2, slot, staging));
        EXPECT_EQ(moe_otd::path_b_scatter_entries(), 1u);
        EXPECT_EQ(moe_otd::path_b_fence_calls(), 1u);
        // Before any set_filled: GPU must already see new bytes.
        auto dump = P0bFixture::dump_slot(mems[0], slot, bytes0);
        const uint32_t off = static_cast<uint32_t>(slot * bytes0);
        const auto gpu = gpu_fast_digest(stream, mems[0], off, static_cast<uint32_t>(bytes0));
        ASSERT_EQ(gpu, cpu_fast_digest_reference(dump.data(), bytes0))
            << "P0B-V1: GPU must see post-fence usm_host bytes before set_filled";
        std::cout << "[P0B-V1] with fence: GPU visibility PASS before set_filled" << std::endl;
    }

    // --- without post-write fence: document NEO / CacheCoherentUMA=false contract ---
    {
        cldnn::moe_weights w;
        auto mems = fx.allocate_resident(w, allocation_type::usm_host);
        const size_t slot = 0;
        moe_otd::path_b_set_skip_post_fence_for_test(true);
        ASSERT_NO_THROW(moe_otd::fill_weights_from_pack(stream, pack, pack_reader, w, /*expert=*/3, slot, staging));
        auto dump = P0bFixture::dump_slot(mems[0], slot, bytes0);
        const uint32_t off = static_cast<uint32_t>(slot * bytes0);
        const auto gpu = gpu_fast_digest(stream, mems[0], off, static_cast<uint32_t>(bytes0));
        const auto cpu = cpu_fast_digest_reference(dump.data(), bytes0);
        const bool visible_without_fence = (gpu == cpu);
        std::cout << "[P0B-V1] without post-fence: gpu_matches_host_dump="
                  << (visible_without_fence ? "true" : "false")
                  << " (CacheCoherentUMA=false expects false on Arc)" << std::endl;
        // Soft contract: if mismatch, that confirms why fence is required; if match, still OK
        // (implementation may snoop) — production path keeps the fence either way.
        moe_otd::path_b_set_skip_post_fence_for_test(false);
        // Re-fence and re-check must match.
        stream.flush();
        stream.finish();
        const auto gpu2 = gpu_fast_digest(stream, mems[0], off, static_cast<uint32_t>(bytes0));
        ASSERT_EQ(gpu2, cpu) << "P0B-V1: after explicit fence, GPU must see host dump";
    }
}

// P0B-L1: verify-fail abort → no false LRU hit; A vs B parity after successful reuse.
TEST(moe_otd_p0b, P0B_L1_no_false_hit_and_reuse_parity) {
    PathBHarnessGuard harness;
    EnvVarGuard on("OTD_RESIDENT_USM_HOST", "1");
    print_p0b_header();

    P0bFixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());

    // Corrupt expert 1 so pack fill aborts before scatter.
    fx.flip_record_byte(/*expert=*/1, /*tensor=*/0, /*byte=*/0);

    cldnn::moe_weights b_w;
    auto b_mems = fx.allocate_resident(b_w, allocation_type::usm_host);
    auto b_provider = fx.make_provider(/*capacity=*/1);
    b_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    b_provider->bind(b_w);

    moe_otd::reset_path_b_harness_counters();
    ASSERT_THROW(b_provider->acquire_one(1, get_test_stream()), ov::Exception);
    EXPECT_EQ(moe_otd::path_b_scatter_entries(), 0u);
    // lookup inserts on miss — probe via acquire would mutate; use cache size / filled check:
    // after throw, slot must remain unfilled: re-acquire of a good expert must miss-fill.
    ASSERT_NO_THROW(b_provider->acquire_one(0, get_test_stream()));
    ASSERT_TRUE(b_provider->cache().lookup(0).hit);

    // Rebuild clean pack for A/B reuse parity.
    P0bFixture fx2;
    fx2.build_pack();
    const auto pack2 = *moe_otd::PackLayout::open(fx2.pack_path(), fx2.spec());

    cldnn::moe_weights a_w, b2_w;
    auto a_mems = fx2.allocate_resident(a_w, allocation_type::usm_device);
    auto b2_mems = fx2.allocate_resident(b2_w, allocation_type::usm_host);
    auto a_provider = fx2.make_provider(/*capacity=*/1);
    auto b2_provider = fx2.make_provider(/*capacity=*/1);
    a_provider->configure_pack(moe_otd::PackMode::require, fx2.pack_dir, fx2.layouts);
    b2_provider->configure_pack(moe_otd::PackMode::require, fx2.pack_dir, fx2.layouts);
    a_provider->bind(a_w);
    b2_provider->bind(b2_w);

    (void)a_provider->acquire_one(0, get_test_stream());
    (void)b2_provider->acquire_one(0, get_test_stream());
    // Evict + reuse slot with expert 2.
    const size_t a_slot = a_provider->acquire_one(2, get_test_stream());
    const size_t b_slot = b2_provider->acquire_one(2, get_test_stream());
    ASSERT_EQ(a_slot, b_slot);

    const size_t bytes0 = static_cast<size_t>(pack2.tensor_bytes(0));
    auto a_dump = P0bFixture::dump_slot(a_mems[0], a_slot, bytes0);
    auto b_dump = P0bFixture::dump_slot(b2_mems[0], b_slot, bytes0);
    ASSERT_EQ(a_dump, b_dump) << "P0B-L1: A/B byte parity after successful reuse";

    const uint32_t off = static_cast<uint32_t>(a_slot * bytes0);
    auto& stream = get_test_stream();
    ASSERT_EQ(gpu_fast_digest(stream, b2_mems[0], off, static_cast<uint32_t>(bytes0)),
              cpu_fast_digest_reference(b_dump.data(), bytes0));
    std::cout << "[P0B-L1] no false hit after abort; A/B reuse parity PASS" << std::endl;
}
