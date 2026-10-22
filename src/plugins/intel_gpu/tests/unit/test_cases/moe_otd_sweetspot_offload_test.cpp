// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Sweet-spot OFFLOAD_RATIO pack suitability (mirrors real C6/Qwen memory_safe setups).
// Production reference: OFFLOAD_RATIO=20 (Qwen memory_safe / C6 argv). Nearby spots: 25, 30
// (functional moe_offload also exercised 25 historically). Pack v1 + require/fill path only;
// not a full-model TTFT bench.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include "ocl_v2/moe/moe_otd_runtime.hpp"
#include "plugin/ops/moe_offload_constant.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
namespace moe_otd = ov::intel_gpu::ocl::moe_otd;

namespace {

// Same capacity math as moe.cpp prepare_moe_otd_params / moe_offload_constant.cpp.
size_t resident_slots_for_ratio(size_t num_expert, size_t offload_ratio) {
    OPENVINO_ASSERT(offload_ratio > 0 && offload_ratio < 100);
    return std::max<size_t>(1, num_expert * (100 - offload_ratio) / 100);
}

struct SweetSpotMoEGraph {
    std::shared_ptr<ov::op::internal::MOECompressed> moe_node;
    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;

    static SweetSpotMoEGraph build(size_t num_experts) {
        SweetSpotMoEGraph g;
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
        inputs.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, hidden_size})->output(0));
        inputs.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, top_k})->output(0));
        inputs.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, top_k})->output(0));

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

// Scaled-down layer: 40 experts mirrors ratio math used on real MoE (capacity differs at 20/25/30).
class SweetSpotPackFixture {
public:
    static constexpr size_t num_expert = 40;
    static constexpr uint8_t sentinel = 0xCD;

    explicit SweetSpotPackFixture(size_t /*offload_ratio*/) {
        std::random_device rd;
        root = std::filesystem::temp_directory_path() / ("moe_otd_sweetspot_" + std::to_string(rd()) + std::to_string(rd()));
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
        for (size_t i = 0; i < bytes.size(); i++) {
            bytes[i] = static_cast<char>((i * 17 + 3) & 0xFF);
        }
        std::ofstream out(weights_path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    ~SweetSpotPackFixture() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    std::unique_ptr<OffloadExpertWeightProvider> make_provider(size_t capacity) const {
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
        auto provider = make_provider(/*capacity=*/2);
        provider->configure_pack(moe_otd::PackMode::build, pack_dir, layouts);
        OPENVINO_ASSERT(provider->uses_pack());
    }

    std::array<memory::ptr, moe_otd::pack_tensor_count> allocate_resident(cldnn::moe_weights& weights,
                                                                           allocation_type alloc = allocation_type::usm_device) const {
        std::array<memory::ptr, moe_otd::pack_tensor_count> mems;
        for (size_t t = 0; t < mems.size(); t++) {
            mems[t] = get_test_engine().allocate_memory(layouts[t], alloc, false);
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

    std::vector<uint8_t> record_bytes(uint32_t expert, size_t tensor) const {
        const auto pack = moe_otd::PackLayout::open(pack_path(), spec());
        OPENVINO_ASSERT(pack.has_value());
        std::ifstream in(pack_path(), std::ios::binary);
        in.seekg(static_cast<std::streamoff>(pack->record_offset(expert) + pack->tensor_offset_in_record(tensor)));
        std::vector<uint8_t> data(pack->tensor_bytes(tensor));
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        return data;
    }

    std::filesystem::path root;
    std::filesystem::path pack_dir;
    std::filesystem::path weights_path;
    MOECompressed::Config config{};
    std::vector<size_t> offsets;
    moe_otd::RoutedTensorLayouts layouts;
};

class SweetSpotOffloadTest : public ::testing::TestWithParam<size_t> {};

}  // namespace

// Formula lock: 20/25/30 match plugin math used by real OVMS/C6 OFFLOAD_RATIO.
TEST_P(SweetSpotOffloadTest, SS01_resident_capacity_matches_plugin_formula) {
    const size_t ratio = GetParam();
    constexpr size_t N = 40;  // scaled MoE layer
    const size_t expect = resident_slots_for_ratio(N, ratio);

    // Hand-checked sweet spots for N=40 (same integer division as moe.cpp).
    if (ratio == 20) {
        EXPECT_EQ(expect, 32u);
    } else if (ratio == 25) {
        EXPECT_EQ(expect, 30u);
    } else if (ratio == 30) {
        EXPECT_EQ(expect, 28u);
    }

    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::offload_ratio(ratio));
    auto graph = SweetSpotMoEGraph::build(N);
    auto constant = graph.constants[0];
    const auto shape = constant->get_output_shape(0);
    const auto layout = cldnn::layout(shape,
                                      cldnn::element_type_to_data_type(constant->get_output_element_type(0)),
                                      cldnn::format::get_default_format(shape.size()));
    const auto desc = ov::intel_gpu::try_prepare_partial_upload(engine,
                                                               config,
                                                               constant,
                                                               shape,
                                                               layout.data_type,
                                                               layout.format,
                                                               layout);
    ASSERT_TRUE(desc.enabled);
    ASSERT_EQ(desc.upload_shape[0], expect);
    std::cout << "[SS01] OFFLOAD_RATIO=" << ratio << " N=" << N << " resident_slots=" << expect
              << " upload_bytes=" << desc.upload_bytes << std::endl;
}

// Pack suitability: build once, require at ratio capacity, serve working set that fits → hits.
TEST_P(SweetSpotOffloadTest, SS02_pack_require_working_set_within_capacity_hits) {
    const size_t ratio = GetParam();
    const size_t capacity = resident_slots_for_ratio(SweetSpotPackFixture::num_expert, ratio);
    SweetSpotPackFixture fx(ratio);
    fx.build_pack();

    auto provider = fx.make_provider(capacity);
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    ASSERT_TRUE(provider->uses_pack());

    cldnn::moe_weights resident;
    const auto mems = fx.allocate_resident(resident);
    provider->bind(resident);

    // Working set = half capacity (fits), like a warm draught of routed experts under memory_safe.
    const size_t ws = std::max<size_t>(2, capacity / 2);
    std::vector<uint32_t> experts;
    for (size_t i = 0; i < ws; i++) {
        experts.push_back(static_cast<uint32_t>(i));
    }

    auto lease = provider->try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value()) << "working set must fit capacity " << capacity << " at ratio " << ratio;
    for (uint32_t e : experts) {
        ASSERT_TRUE(provider->cache().lookup(e).hit);
        const size_t slot = provider->cache().lookup(e).slot;
        const auto expected = fx.record_bytes(e, /*tensor=*/0);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock(mems[0], get_test_stream());
        for (size_t i = 0; i < expected.size(); i++) {
            ASSERT_EQ(lock[slot * expected.size() + i], expected[i]) << "ratio=" << ratio << " expert=" << e;
        }
    }

    // Second simultaneous acquire: all hits (no extra miss fills needed for correctness claim).
    auto lease2 = provider->try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease2.has_value());
    ASSERT_EQ(*lease, *lease2);

    std::cout << "[SS02] OFFLOAD_RATIO=" << ratio << " capacity=" << capacity << " working_set=" << ws
              << " pack_require+hits OK" << std::endl;
}

// Over-capacity simultaneous reject (Wave-01) + sequential acquire still serves via eviction.
TEST_P(SweetSpotOffloadTest, SS03_over_capacity_rejects_simultaneous_sequential_evicts_ok) {
    const size_t ratio = GetParam();
    const size_t capacity = resident_slots_for_ratio(SweetSpotPackFixture::num_expert, ratio);
    SweetSpotPackFixture fx(ratio);
    fx.build_pack();

    auto provider = fx.make_provider(capacity);
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    ASSERT_TRUE(provider->uses_pack());
    cldnn::moe_weights resident;
    fx.allocate_resident(resident);
    provider->bind(resident);

    std::vector<uint32_t> too_many;
    for (size_t i = 0; i < capacity + 1; i++) {
        too_many.push_back(static_cast<uint32_t>(i));
    }
    ASSERT_FALSE(provider->try_acquire_simultaneous(too_many, get_test_stream()).has_value());

    // Sequential path: fill capacity without probing cache (lookup mutates LRU), then overflow.
    for (size_t i = 0; i < capacity; i++) {
        provider->acquire_one(static_cast<uint32_t>(i), get_test_stream());
    }
    provider->acquire_one(static_cast<uint32_t>(capacity), get_test_stream());
    // Prove eviction via expert_loads on re-acquire of oldest (0), not via lookup-on-miss.
    auto* perf = moe_otd::get_perf_counters();
    const uint64_t before = perf ? perf->expert_loads.load() : 0;
    provider->acquire_one(0, get_test_stream());
    if (perf) {
        EXPECT_GE(perf->expert_loads.load(), before + 1);
    }

    std::cout << "[SS03] OFFLOAD_RATIO=" << ratio << " capacity=" << capacity
              << " overflow reject + sequential eviction OK" << std::endl;
}

// Sweet-spot path B (UMA scratch): ratio=20 pack serve byte-parity vs path A.
TEST(moe_otd_sweetspot, SS04_ratio20_path_b_pack_parity_with_path_a) {
    constexpr size_t ratio = 20;
    const size_t capacity = resident_slots_for_ratio(SweetSpotPackFixture::num_expert, ratio);
    ASSERT_EQ(capacity, 32u);

    SweetSpotPackFixture fx(ratio);
    fx.build_pack();
    const auto pack = *moe_otd::PackLayout::open(fx.pack_path(), fx.spec());

    cldnn::moe_weights a_w, b_w;
    auto a_mems = fx.allocate_resident(a_w, allocation_type::usm_device);
    auto b_mems = fx.allocate_resident(b_w, allocation_type::usm_host);

    auto a_provider = fx.make_provider(capacity);
    a_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    a_provider->bind(a_w);
    auto b_provider = fx.make_provider(capacity);
    b_provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    b_provider->bind(b_w);

    const uint32_t expert = 17;  // mid-range expert under memory_safe-ish working set
    const size_t a_slot = a_provider->acquire_one(expert, get_test_stream());
    const size_t b_slot = b_provider->acquire_one(expert, get_test_stream());
    ASSERT_EQ(a_slot, b_slot);

    for (size_t t = 0; t < moe_otd::pack_tensor_count; t++) {
        const size_t bytes = static_cast<size_t>(pack.tensor_bytes(t));
        if (bytes == 0) {
            continue;
        }
        cldnn::mem_lock<uint8_t, mem_lock_type::read> la(a_mems[t], get_test_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lb(b_mems[t], get_test_stream());
        for (size_t i = 0; i < bytes; i++) {
            ASSERT_EQ(la[a_slot * bytes + i], lb[b_slot * bytes + i]) << "tensor " << t << " byte " << i;
        }
    }
    std::cout << "[SS04] OFFLOAD_RATIO=20 capacity=32 path A/B pack parity OK (UMA scratch)" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(OffloadSweetSpots,
                         SweetSpotOffloadTest,
                         ::testing::Values(20, 25, 30),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                             return "ratio_" + std::to_string(info.param);
                         });
