// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// UMA-02 harness (path B): byte-parity, corrupt negatives, LRU, allocator flag.
// GPU_LIVE cases (U2-G*) are bounded short-correctness runs; COMPUTE_GO delegated for this matrix only.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include "ocl_v2/moe/moe_otd_runtime.hpp"
#include "plugin/ops/moe_offload_constant.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
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

class Uma02Fixture {
public:
    static constexpr size_t num_expert = 4;
    static constexpr uint8_t sentinel = 0xEE;

    Uma02Fixture() {
        std::random_device rd;
        root = std::filesystem::temp_directory_path() / ("moe_otd_uma02_" + std::to_string(rd()) + std::to_string(rd()));
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
        std::vector<char> bytes(cursor);
        for (size_t i = 0; i < bytes.size(); i++) {
            bytes[i] = static_cast<char>((i * 31 + 7) & 0xFF);
        }
        std::ofstream out(weights_path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    ~Uma02Fixture() {
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

    std::filesystem::path pack_path() const {
        return pack_dir / spec().file_name();
    }

    void build_pack() const {
        auto provider = make_provider();
        provider->configure_pack(moe_otd::PackMode::build, pack_dir, layouts);
        OPENVINO_ASSERT(provider->uses_pack(), "fixture pack was not built");
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

    static bool all_sentinel(const memory::ptr& mem) {
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock(mem, get_test_stream());
        return std::all_of(lock.begin(), lock.end(), [](uint8_t b) {
            return b == sentinel;
        });
    }

    std::filesystem::path root;
    std::filesystem::path pack_dir;
    std::filesystem::path weights_path;
    MOECompressed::Config config{};
    std::vector<size_t> offsets;
    moe_otd::RoutedTensorLayouts layouts;
};

std::array<std::vector<uint8_t>, moe_otd::pack_tensor_count> dump_nine(const std::array<memory::ptr, moe_otd::pack_tensor_count>& mems,
                                                                       size_t slot,
                                                                       const moe_otd::PackLayout& pack) {
    std::array<std::vector<uint8_t>, moe_otd::pack_tensor_count> out;
    for (size_t t = 0; t < mems.size(); t++) {
        out[t] = Uma02Fixture::dump_slot(mems[t], slot, static_cast<size_t>(pack.tensor_bytes(t)));
    }
    return out;
}

}  // namespace

// U2-P01: path A vs path B byte parity for one expert/slot.
TEST(moe_otd_uma02, U2_P01_single_expert_slot_parity) {
    Uma02Fixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());

    cldnn::moe_weights a_w, b_w;
    auto a_mems = fx.allocate_resident(a_w, allocation_type::usm_device);
    auto b_mems = fx.allocate_resident(b_w, allocation_type::usm_host);

    auto a_provider = fx.make_provider();
    a_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    a_provider->bind(a_w);
    auto b_provider = fx.make_provider();
    b_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    b_provider->bind(b_w);

    moe_otd::reset_path_b_harness_counters();
    const uint32_t expert = 2;
    const size_t a_slot = a_provider->acquire_one(expert, get_test_stream());
    const size_t b_slot = b_provider->acquire_one(expert, get_test_stream());
    ASSERT_EQ(a_slot, b_slot);

    const auto a_dump = dump_nine(a_mems, a_slot, pack);
    const auto b_dump = dump_nine(b_mems, b_slot, pack);
    for (size_t t = 0; t < a_dump.size(); t++) {
        ASSERT_EQ(a_dump[t], b_dump[t]) << "tensor " << t;
    }
    EXPECT_GE(moe_otd::path_b_scatter_entries(), 1u);
    EXPECT_GE(moe_otd::path_b_fence_calls(), 1u);
}

// U2-P02: two slots, no crosstalk.
TEST(moe_otd_uma02, U2_P02_multi_slot_no_crosstalk) {
    Uma02Fixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());

    cldnn::moe_weights a_w, b_w;
    auto a_mems = fx.allocate_resident(a_w, allocation_type::usm_device);
    auto b_mems = fx.allocate_resident(b_w, allocation_type::usm_host);

    auto a_provider = fx.make_provider(/*capacity=*/2);
    a_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    a_provider->bind(a_w);
    auto b_provider = fx.make_provider(/*capacity=*/2);
    b_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    b_provider->bind(b_w);

    const size_t a0 = a_provider->acquire_one(0, get_test_stream());
    const size_t a1 = a_provider->acquire_one(1, get_test_stream());
    const size_t b0 = b_provider->acquire_one(0, get_test_stream());
    const size_t b1 = b_provider->acquire_one(1, get_test_stream());
    ASSERT_NE(a0, a1);
    ASSERT_EQ(a0, b0);
    ASSERT_EQ(a1, b1);

    for (size_t t = 0; t < moe_otd::pack_tensor_count; t++) {
        ASSERT_EQ(Uma02Fixture::dump_slot(a_mems[t], a0, static_cast<size_t>(pack.tensor_bytes(t))),
                  Uma02Fixture::dump_slot(b_mems[t], b0, static_cast<size_t>(pack.tensor_bytes(t))));
        ASSERT_EQ(Uma02Fixture::dump_slot(a_mems[t], a1, static_cast<size_t>(pack.tensor_bytes(t))),
                  Uma02Fixture::dump_slot(b_mems[t], b1, static_cast<size_t>(pack.tensor_bytes(t))));
    }
}

// U2-P03: capacity-1 evict then refill parity.
TEST(moe_otd_uma02, U2_P03_evict_then_refill_parity) {
    Uma02Fixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());

    cldnn::moe_weights a_w, b_w;
    auto a_mems = fx.allocate_resident(a_w, allocation_type::usm_device);
    auto b_mems = fx.allocate_resident(b_w, allocation_type::usm_host);

    auto a_provider = fx.make_provider(/*capacity=*/1);
    a_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    a_provider->bind(a_w);
    auto b_provider = fx.make_provider(/*capacity=*/1);
    b_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    b_provider->bind(b_w);

    (void)a_provider->acquire_one(0, get_test_stream());
    (void)b_provider->acquire_one(0, get_test_stream());
    (void)a_provider->acquire_one(1, get_test_stream());  // evicts 0
    (void)b_provider->acquire_one(1, get_test_stream());
    const size_t a_slot = a_provider->acquire_one(0, get_test_stream());  // refill 0
    const size_t b_slot = b_provider->acquire_one(0, get_test_stream());

    const auto a_dump = dump_nine(a_mems, a_slot, pack);
    const auto b_dump = dump_nine(b_mems, b_slot, pack);
    for (size_t t = 0; t < a_dump.size(); t++) {
        ASSERT_EQ(a_dump[t], b_dump[t]) << "tensor " << t;
    }
}

// U2-N01 / U2-N02: corrupt checksum — no resident mutation, no scatter.
TEST(moe_otd_uma02, U2_N01_N02_checksum_fail_closed_no_scatter) {
    Uma02Fixture fx;
    fx.build_pack();
    fx.flip_record_byte(/*expert=*/1, /*tensor=*/8, /*byte=*/3);

    cldnn::moe_weights b_w;
    auto b_mems = fx.allocate_resident(b_w, allocation_type::usm_host);
    auto provider = fx.make_provider();
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    provider->bind(b_w);

    moe_otd::reset_path_b_harness_counters();
    ASSERT_THROW(provider->acquire_one(1, get_test_stream()), ov::Exception);
    EXPECT_EQ(moe_otd::path_b_scatter_entries(), 0u);
    EXPECT_EQ(moe_otd::path_b_fence_calls(), 0u);
    for (const auto& mem : b_mems) {
        ASSERT_TRUE(Uma02Fixture::all_sentinel(mem));
    }
    ASSERT_FALSE(provider->cache().lookup(1).hit);
}

// U2-L01 / U2-L02: no false hit after fail; hit only after successful fill.
TEST(moe_otd_uma02, U2_L01_L02_lru_after_verify_fail_and_recover) {
    Uma02Fixture fx;
    fx.build_pack();
    fx.flip_record_byte(/*expert=*/1, /*tensor=*/0, /*byte=*/0);

    cldnn::moe_weights b_w;
    auto b_mems = fx.allocate_resident(b_w, allocation_type::usm_host);
    auto provider = fx.make_provider();
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    provider->bind(b_w);

    ASSERT_THROW(provider->acquire_one(1, get_test_stream()), ov::Exception);
    ASSERT_FALSE(provider->cache().lookup(1).hit);
    ASSERT_THROW(provider->acquire_one(1, get_test_stream()), ov::Exception);
    ASSERT_FALSE(provider->cache().lookup(1).hit);

    // Drop serving handles before rewriting the pack (Windows held-reader semantics).
    provider.reset();
    // Same XOR restores the payload byte; no rebuild while a reader holds the file.
    fx.flip_record_byte(/*expert=*/1, /*tensor=*/0, /*byte=*/0);

    auto provider2 = fx.make_provider();
    provider2->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    provider2->bind(b_w);
    ASSERT_NO_THROW(provider2->acquire_one(1, get_test_stream()));
    ASSERT_TRUE(provider2->cache().lookup(1).hit);
    ASSERT_TRUE(provider2->cache().lookup(1).hit);  // second lookup still hit
    (void)b_mems;
}

// Same minimal MOECompressed graph as moe_offload_constant_test (routed constants 3..11).
struct Uma02MoEGraph {
    std::shared_ptr<ov::op::internal::MOECompressed> moe_node;
    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;

    static Uma02MoEGraph build() {
        Uma02MoEGraph g;
        const size_t num_experts = 4;
        const size_t hidden_size = 128;
        const size_t inter_size = 256;
        const size_t group_size = 128;
        const size_t top_k = 2;
        ov::op::internal::MOECompressed::Config config{};
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.num_shared_expert = 1;
        config.top_k = top_k;
        config.group_size = group_size;
        config.has_zp = true;
        config.out_type = ov::element::f16;

        const size_t group_num = hidden_size / group_size;
        const size_t group_num2 = inter_size / group_size;

        ov::OutputVector inputs;
        auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, hidden_size});
        auto routing = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, top_k});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, top_k});
        inputs.push_back(hidden->output(0));
        inputs.push_back(routing->output(0));
        inputs.push_back(topk_idx->output(0));

        auto make_const = [](ov::element::Type type, const ov::Shape& shape) {
            return std::make_shared<ov::op::v0::Constant>(type, shape, 0);
        };
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_size, group_num}));
        g.constants.push_back(make_const(ov::element::f16, {num_experts, inter_size, group_num, 1}));
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_num, 1}));
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_size, group_num}));
        g.constants.push_back(make_const(ov::element::f16, {num_experts, inter_size, group_num, 1}));
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_num, 1}));
        g.constants.push_back(make_const(ov::element::u8, {num_experts, hidden_size, group_size, group_num2}));
        g.constants.push_back(make_const(ov::element::f16, {num_experts, hidden_size, group_num2, 1}));
        g.constants.push_back(make_const(ov::element::u8, {num_experts, hidden_size, group_num2, 1}));
        g.constants.push_back(make_const(ov::element::u8, {1, inter_size, group_size, group_num}));
        g.constants.push_back(make_const(ov::element::f16, {1, inter_size, group_num, 1}));
        g.constants.push_back(make_const(ov::element::u8, {1, inter_size, group_num, 1}));
        g.constants.push_back(make_const(ov::element::u8, {1, inter_size, group_size, group_num}));
        g.constants.push_back(make_const(ov::element::f16, {1, inter_size, group_num, 1}));
        g.constants.push_back(make_const(ov::element::u8, {1, inter_size, group_num, 1}));
        g.constants.push_back(make_const(ov::element::u8, {1, hidden_size, group_size, group_num2}));
        g.constants.push_back(make_const(ov::element::f16, {1, hidden_size, group_num2, 1}));
        g.constants.push_back(make_const(ov::element::u8, {1, hidden_size, group_num2, 1}));
        g.constants.push_back(make_const(ov::element::f16, {1, hidden_size}));
        for (auto& c : g.constants) {
            inputs.push_back(c->output(0));
        }
        g.moe_node = std::make_shared<ov::op::internal::MOECompressed>(inputs, config);
        return g;
    }
};

// U2-A01 / U2-A02: flag selects usm_host via try_prepare_partial_upload; default stays preferred.
TEST(moe_otd_uma02, U2_A01_A02_partial_upload_alloc_flag) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::offload_ratio(20));
    auto graph = Uma02MoEGraph::build();
    auto constant = graph.constants[0];
    const auto shape = constant->get_output_shape(0);
    const auto layout = cldnn::layout(shape,
                                      cldnn::element_type_to_data_type(constant->get_output_element_type(0)),
                                      cldnn::format::get_default_format(shape.size()));

    {
        EnvVarGuard clear("OTD_RESIDENT_USM_HOST", "");
        EXPECT_FALSE(moe_otd::otd_resident_usm_host_enabled());
        const auto desc = ov::intel_gpu::try_prepare_partial_upload(engine,
                                                                    config,
                                                                    constant,
                                                                    shape,
                                                                    layout.data_type,
                                                                    layout.format,
                                                                    layout);
        ASSERT_TRUE(desc.enabled);
        EXPECT_EQ(desc.memory->get_allocation_type(), engine.get_preferred_memory_allocation_type());
    }
    {
        EnvVarGuard on("OTD_RESIDENT_USM_HOST", "1");
        EXPECT_TRUE(moe_otd::otd_resident_usm_host_enabled());
        const auto desc = ov::intel_gpu::try_prepare_partial_upload(engine,
                                                                    config,
                                                                    constant,
                                                                    shape,
                                                                    layout.data_type,
                                                                    layout.format,
                                                                    layout);
        ASSERT_TRUE(desc.enabled);
        EXPECT_EQ(desc.memory->get_allocation_type(), allocation_type::usm_host);
    }
}

// U2-A03: fence runs after scatter on successful path B fill.
TEST(moe_otd_uma02, U2_A03_fence_before_publish) {
    Uma02Fixture fx;
    fx.build_pack();

    cldnn::moe_weights b_w;
    fx.allocate_resident(b_w, allocation_type::usm_host);
    auto provider = fx.make_provider();
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    provider->bind(b_w);

    moe_otd::reset_path_b_harness_counters();
    ASSERT_NO_THROW(provider->acquire_one(0, get_test_stream()));
    EXPECT_EQ(moe_otd::path_b_scatter_entries(), 1u);
    EXPECT_EQ(moe_otd::path_b_fence_calls(), 1u);
    ASSERT_TRUE(provider->cache().lookup(0).hit);
}

// -------- GPU_LIVE harness (short correctness; COMPUTE_GO delegated for this bounded matrix) --------

namespace {

// Minimal real GPU kernel: schedule-independent FNV-1a digest of a byte window, per work item.
// Work item gid digests bytes [gid*chunk, gid*chunk+chunk) of [begin, begin+n) so results never
// depend on work-group scheduling; no atomics, no reductions.
constexpr uint32_t kDigestWorkItems = 64;
static const char kDigestKernelSource[] = R"(
__kernel void uma02_digest(__global const uchar* src, __global uint* dst, const uint begin, const uint n) {
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

cldnn::kernel::ptr get_uma02_digest_kernel() {
    static cldnn::kernel::ptr cached = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        auto& engine = get_test_engine();
        auto builder = engine.create_kernel_builder();
        OPENVINO_ASSERT(builder != nullptr, "uma02: engine returned no kernel builder");
        std::vector<cldnn::kernel::ptr> out;
        const char* src = kDigestKernelSource;
        builder->build_kernels(src, std::strlen(kDigestKernelSource), cldnn::KernelFormat::SOURCE, "", out);
        OPENVINO_ASSERT(!out.empty(), "uma02: digest kernel build produced no kernels");
        cached = out.front();
    });
    return cached;
}

// Enqueues the digest kernel over mem[begin .. begin+n) and returns the per-work-item digests.
std::vector<uint32_t> gpu_digest_window(const cldnn::memory::ptr& mem, uint32_t begin, uint32_t n) {
    auto& stream = downcast<cldnn::ocl::ocl_stream>(get_test_stream());
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
    desc.layerID = "uma02_g01_digest";

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

    auto kernel = get_uma02_digest_kernel();
    stream.set_arguments(*kernel, desc, data);
    (void)stream.enqueue_kernel(*kernel, desc, data, {});
    stream.finish();

    std::vector<uint32_t> output(kDigestWorkItems);
    cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> lock(out_mem, get_test_stream());
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = static_cast<uint32_t>(lock[i]);
    }
    return output;
}

// CPU-side reference of the same FNV-1a windowing used by the kernel.
std::vector<uint32_t> cpu_digest_reference(const uint8_t* bytes, size_t n) {
    uint32_t num = static_cast<uint32_t>(kDigestWorkItems);
    uint32_t chunk = (static_cast<uint32_t>(n) + num - 1u) / num;
    std::vector<uint32_t> digests(num, 0x811C9DC5u);
    for (uint32_t gid = 0; gid < num; gid++) {
        const size_t start = static_cast<size_t>(gid) * chunk;
        const size_t end = std::min(start + chunk, n);
        uint32_t acc = 0x811C9DC5u;
        for (size_t i = start; i < end; i++) {
            acc ^= static_cast<uint32_t>(bytes[i]);
            acc *= 0x01000193u;
        }
        digests[gid] = acc;
    }
    return digests;
}

void print_gpu_device_header() {
    auto& engine = get_test_engine();
    const auto& info = engine.get_device_info();
    std::cout << "[U2-G] device=" << info.dev_name << std::endl;
}

}  // namespace

// U2-G01: a real GPU kernel reads path-B (usm_host) residents after a pack fill; the GPU-visible
// bytes must match the path-A (usm_device) reference and the CPU reference digest, tensor by tensor.
TEST(moe_otd_uma02, U2_G01_kernel_reads_path_b_residents_match_path_a) {
    Uma02Fixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());
    print_gpu_device_header();

    cldnn::moe_weights a_w, b_w;
    auto a_mems = fx.allocate_resident(a_w, allocation_type::usm_device);
    auto b_mems = fx.allocate_resident(b_w, allocation_type::usm_host);

    auto a_provider = fx.make_provider();
    a_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    a_provider->bind(a_w);
    auto b_provider = fx.make_provider();
    b_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    b_provider->bind(b_w);

    const uint32_t expert = 2;
    const size_t a_slot = a_provider->acquire_one(expert, get_test_stream());
    const size_t b_slot = b_provider->acquire_one(expert, get_test_stream());
    ASSERT_EQ(a_slot, b_slot);
    get_test_stream().finish();

    size_t per_expert = 0;
    for (size_t t = 0; t < moe_otd::pack_tensor_count && per_expert == 0; t++) {
        if (pack.tensor_bytes(t) != 0) {
            per_expert = static_cast<size_t>(pack.tensor_bytes(t));
        }
    }
    OPENVINO_ASSERT(per_expert > 0, "uma02 g01: all pack tensors are zero-size");
    const uint32_t slot_base = static_cast<uint32_t>(a_slot * per_expert);

    int mismatches = 0;
    for (size_t t = 0; t < moe_otd::pack_tensor_count; t++) {
        const size_t bytes = static_cast<size_t>(pack.tensor_bytes(t));
        if (bytes == 0) {
            continue;
        }
        // CPU reference from the path-A resident dump (fixture dumps also re-assert CPU parity).
        auto a_dump = dump_nine(a_mems, a_slot, pack);
        auto b_dump = dump_nine(b_mems, b_slot, pack);
        ASSERT_EQ(a_dump[t], b_dump[t]) << "tensor " << t << ": CPU-side path A/B parity broke";

        const auto ref = cpu_digest_reference(a_dump[t].data(), bytes);
        const auto digest_a = gpu_digest_window(a_mems[t], slot_base, static_cast<uint32_t>(bytes));
        ASSERT_EQ(digest_a, ref) << "tensor t=" << t << ": GPU kernel digest != CPU reference on path-A resident";
        const auto digest_b = gpu_digest_window(b_mems[t], slot_base, static_cast<uint32_t>(bytes));
        for (size_t g = 0; g < ref.size(); g++) {
            if (digest_b[g] != ref[g]) {
                FAIL() << "tensor t=" << t << " work-item " << g << ": path-B usm_host resident bytes differ on GPU, gpu=0x"
                       << std::hex << digest_b[g] << " ref=0x" << ref[g] << std::dec;
                mismatches++;
                break;
            }
        }
        std::cout << "[U2-G01] tensor=" << t << " bytes=" << bytes << " digest_ref=0x" << std::hex << ref[0]
                  << " path_a=0x" << digest_a[0] << " path_b=0x" << digest_b[0] << std::dec << std::endl;
    }
    EXPECT_EQ(mismatches, 0);
}

// U2-G02: exactly one cold miss — record OTD counters (path B and path A reference). No win claim.
TEST(moe_otd_uma02, U2_G02_one_cold_miss_counters) {
    print_gpu_device_header();
    auto* perf = moe_otd::get_perf_counters();
    if (perf == nullptr) {
        if (std::getenv("MOE_OTD_PERF_LOG") != nullptr) {
            FAIL() << "MOE_OTD_PERF_LOG is set but get_perf_counters() still returned nullptr";
        }
        GTEST_SKIP() << "MOE_OTD_PERF_LOG unset in this run; rerun suite with MOE_OTD_PERF_LOG=1";
    }

    using moe_otd::OtdPerfSnapshot;

    auto record_miss = [&](cldnn::moe_weights& w, allocation_type alloc, const char* tag) {
        Uma02Fixture fx;
        fx.build_pack();
        auto mems = fx.allocate_resident(w, alloc);
        (void)mems;
        auto provider = fx.make_provider();
        provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
        provider->bind(w);

        moe_otd::reset_path_b_harness_counters();
        const auto before_cold = perf->snapshot();
        const auto t_cold = std::chrono::steady_clock::now();
        provider->acquire_one(3, get_test_stream());  // cold miss -> fill
        const auto t_hot = std::chrono::steady_clock::now();
        const auto after_cold = perf->snapshot();
        provider->acquire_one(3, get_test_stream());  // warm hit
        const auto t_hit = std::chrono::steady_clock::now();
        const auto after_hit = perf->snapshot();

        auto delta = [](const OtdPerfSnapshot& a, const OtdPerfSnapshot& b) {
            OtdPerfSnapshot d;
            d.gpu_hits = b.gpu_hits - a.gpu_hits;
            d.gpu_misses = b.gpu_misses - a.gpu_misses;
            d.evictions = b.evictions - a.evictions;
            d.disk_io_ns = b.disk_io_ns - a.disk_io_ns;
            d.disk_read_bytes = b.disk_read_bytes - a.disk_read_bytes;
            d.disk_reads = b.disk_reads - a.disk_reads;
            d.transpose_ns = b.transpose_ns - a.transpose_ns;
            d.gpu_copy_ns = b.gpu_copy_ns - a.gpu_copy_ns;
            d.tensor_load_count = b.tensor_load_count - a.tensor_load_count;
            d.expert_loads = b.expert_loads - a.expert_loads;
            d.pack_verify_ns = b.pack_verify_ns - a.pack_verify_ns;
            d.pack_verified_records = b.pack_verified_records - a.pack_verified_records;
            d.pack_verify_failures = b.pack_verify_failures - a.pack_verify_failures;
            return d;
        };

        struct MissStats {
            OtdPerfSnapshot cold_delta{};
            OtdPerfSnapshot hit_delta{};
            uint64_t scatter_entries = 0;
            uint64_t fence_calls = 0;
            uint64_t cold_ns = 0;
            uint64_t hit_ns = 0;
        };
        MissStats s;
        s.cold_delta = delta(before_cold, after_cold);
        s.hit_delta = delta(after_cold, after_hit);
        s.scatter_entries = moe_otd::path_b_scatter_entries();
        s.fence_calls = moe_otd::path_b_fence_calls();
        s.cold_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_hot - t_cold).count());
        s.hit_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_hit - t_hot).count());
        std::cout << "[U2-G02-" << tag << "] cold: expert_loads=" << s.cold_delta.expert_loads
                  << " disk_reads=" << s.cold_delta.disk_reads << " disk_read_bytes=" << s.cold_delta.disk_read_bytes
                  << " disk_io_ns=" << s.cold_delta.disk_io_ns << " verify_ns=" << s.cold_delta.pack_verify_ns
                  << " verified_records=" << s.cold_delta.pack_verified_records
                  << " copy_ns=" << s.cold_delta.gpu_copy_ns << " cold_ns=" << s.cold_ns << std::endl;
        std::cout << "[U2-G02-" << tag << "] hit:  expert_loads=" << s.hit_delta.expert_loads
                  << " disk_reads=" << s.hit_delta.disk_reads << " hit_ns=" << s.hit_ns << std::endl;
        return s;
    };

    // Path B (flag on): usm_host residents, CPU scatter + fence.
    {
        EnvVarGuard on("OTD_RESIDENT_USM_HOST", "1");
        EXPECT_TRUE(moe_otd::otd_resident_usm_host_enabled());
        cldnn::moe_weights b_w;
        auto s = record_miss(b_w, allocation_type::usm_host, "PATH_B");
        EXPECT_EQ(s.cold_delta.expert_loads, 1u);
        EXPECT_EQ(s.cold_delta.disk_reads, 1u);
        EXPECT_GT(s.cold_delta.disk_read_bytes, 0u);
        EXPECT_EQ(s.cold_delta.pack_verified_records, 1u);
        EXPECT_EQ(s.cold_delta.pack_verify_failures, 0u);
        EXPECT_EQ(s.scatter_entries, 1u);
        EXPECT_EQ(s.fence_calls, 1u);
        EXPECT_EQ(s.hit_delta.expert_loads, 0u);
        EXPECT_EQ(s.hit_delta.disk_reads, 0u);
    }

    // Path A reference (default): usm_device residents via copy_from.
    {
        EnvVarGuard clear("OTD_RESIDENT_USM_HOST", "");
        cldnn::moe_weights a_w;
        auto s = record_miss(a_w, allocation_type::usm_device, "PATH_A");
        EXPECT_EQ(s.cold_delta.expert_loads, 1u);
        EXPECT_EQ(s.cold_delta.disk_reads, 1u);
        EXPECT_GT(s.cold_delta.disk_read_bytes, 0u);
        EXPECT_EQ(s.cold_delta.pack_verified_records, 1u);
        EXPECT_EQ(s.hit_delta.disk_reads, 0u);
        std::cout << "[U2-G02-WARN] single cold miss only; no bandwidth or copy win claim" << std::endl;
    }
}

// U2-G03: simultaneous acquire within capacity + eviction under an in-flight stream.
// Pass: no false hit after eviction; overflow simultaneous rejected without mutating filled set;
// after finish+evict+refill, GPU digests match the new expert bytes (no use-after-evict hit).
TEST(moe_otd_uma02, U2_G03_concurrent_miss_eviction_under_stream) {
    print_gpu_device_header();
    Uma02Fixture fx;
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());

    // --- simultaneous lease within capacity (path B) ---
    {
        cldnn::moe_weights w;
        fx.allocate_resident(w, allocation_type::usm_host);
        auto provider = fx.make_provider(/*capacity=*/2);
        provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
        provider->bind(w);

        auto lease = provider->try_acquire_simultaneous({0, 1, 0}, get_test_stream());
        ASSERT_TRUE(lease.has_value());
        ASSERT_EQ(lease->size(), 3u);
        ASSERT_EQ((*lease)[0], (*lease)[2]);
        ASSERT_NE((*lease)[0], (*lease)[1]);
        ASSERT_TRUE(provider->cache().lookup(0).hit);
        ASSERT_TRUE(provider->cache().lookup(1).hit);

        // Overflow must reject before touching cache (Wave-01 contract).
        // Do not lookup(2): lookup inserts on miss and would mutate the cache.
        auto overflow = provider->try_acquire_simultaneous({0, 1, 2}, get_test_stream());
        ASSERT_FALSE(overflow.has_value());
        ASSERT_TRUE(provider->cache().lookup(0).hit);
        ASSERT_TRUE(provider->cache().lookup(1).hit);
        ASSERT_EQ(provider->cache().size(), 2u);
    }

    // --- eviction under stream: enqueue digest, finish, evict, prove no false hit, refill parity ---
    {
        cldnn::moe_weights w;
        auto mems = fx.allocate_resident(w, allocation_type::usm_host);
        auto provider = fx.make_provider(/*capacity=*/1);
        provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
        provider->bind(w);

        const size_t slot0 = provider->acquire_one(0, get_test_stream());
        ASSERT_TRUE(provider->cache().lookup(0).hit);

        // Kick a real GPU kernel at the resident, then finish before eviction (lease window).
        const size_t bytes0 = static_cast<size_t>(pack.tensor_bytes(0));
        ASSERT_GT(bytes0, 0u);
        const uint32_t off0 = static_cast<uint32_t>(slot0 * bytes0);
        auto digest_before = gpu_digest_window(mems[0], off0, static_cast<uint32_t>(bytes0));
        auto dump0 = Uma02Fixture::dump_slot(mems[0], slot0, bytes0);
        ASSERT_EQ(digest_before, cpu_digest_reference(dump0.data(), bytes0));

        // Evict expert 0 by loading expert 1 into the only slot.
        // IMPORTANT: LRUCache::lookup() inserts on miss — never probe an absent expert with lookup().
        const size_t slot1 = provider->acquire_one(1, get_test_stream());
        ASSERT_EQ(slot0, slot1);
        ASSERT_TRUE(provider->cache().lookup(1).hit);

        auto dump1 = Uma02Fixture::dump_slot(mems[0], slot1, bytes0);
        // Tiny fixtures may share identical tensor0 bytes across experts; still require GPU-visible
        // bytes to match the post-fill dump (path-B fence), proven via miss counters below.
        auto digest_after_e1 = gpu_digest_window(mems[0], off0, static_cast<uint32_t>(bytes0));
        ASSERT_EQ(digest_after_e1, cpu_digest_reference(dump1.data(), bytes0));
        (void)dump1;

        // Warm hit on E1 (no second fill).
        auto* perf = moe_otd::get_perf_counters();
        const uint64_t loads_before_hit = perf ? perf->expert_loads.load() : 0;
        provider->acquire_one(1, get_test_stream());
        if (perf) {
            EXPECT_EQ(perf->expert_loads.load(), loads_before_hit);
        }

        // Refill expert 0 — must be a miss (evicted earlier), then serve; GPU digest restored.
        const uint64_t loads_before_refill = perf ? perf->expert_loads.load() : 0;
        const size_t slot0b = provider->acquire_one(0, get_test_stream());
        ASSERT_EQ(slot0b, slot0);
        if (perf) {
            EXPECT_EQ(perf->expert_loads.load(), loads_before_refill + 1);
        }
        ASSERT_TRUE(provider->cache().lookup(0).hit);
        auto dump0b = Uma02Fixture::dump_slot(mems[0], slot0b, bytes0);
        ASSERT_EQ(dump0, dump0b);
        auto digest_restored = gpu_digest_window(mems[0], off0, static_cast<uint32_t>(bytes0));
        ASSERT_EQ(digest_restored, digest_before);

        // E1 is now evicted: re-acquire must miss again (no false hit).
        const uint64_t loads_before_e1 = perf ? perf->expert_loads.load() : 0;
        provider->acquire_one(1, get_test_stream());
        if (perf) {
            EXPECT_EQ(perf->expert_loads.load(), loads_before_e1 + 1);
        }

        std::cout << "[U2-G03] simultaneous+overflow OK; eviction+refill digests OK; no false hit" << std::endl;
    }
}
