// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Host-only UMA-02 harness (path B): byte-parity, corrupt negatives, LRU, allocator flag.
// GPU_LIVE cases (U2-G*) are intentionally absent — COMPUTE_GO not granted.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
