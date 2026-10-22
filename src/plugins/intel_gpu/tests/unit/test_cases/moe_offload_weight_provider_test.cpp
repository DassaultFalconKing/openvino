// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <random>

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
using ov::intel_gpu::ocl::moe::ResidentExpertWeightProvider;
namespace moe_otd = ov::intel_gpu::ocl::moe_otd;

namespace {

// A tiny routed-expert layer (4 experts, u8 weights, f16 scales, u8 zero points) whose weights
// file and layouts satisfy the same scale/zp transpose that build_layer_pack applies.
class OtdPackLayerFixture {
public:
    static constexpr size_t num_expert = 4;
    static constexpr uint8_t sentinel = 0xEE;

    OtdPackLayerFixture() {
        std::random_device rd;
        root = std::filesystem::temp_directory_path() / ("moe_otd_provider_test_" + std::to_string(rd()) + std::to_string(rd()));
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
            layout{ov::PartialShape{e, 32, gate_groups, 32}, data_types::u8, format::bfyx},   // gate_w
            layout{ov::PartialShape{e, 32, gate_groups, 32}, data_types::u8, format::bfyx},   // up_w
            layout{ov::PartialShape{e, 64, down_groups, 32}, data_types::u8, format::bfyx},   // down_w
            layout{ov::PartialShape{e, 32, gate_groups, 1}, data_types::f16, format::bfyx},   // gate_s
            layout{ov::PartialShape{e, 32, gate_groups, 1}, data_types::f16, format::bfyx},   // up_s
            layout{ov::PartialShape{e, 64, down_groups, 1}, data_types::f16, format::bfyx},   // down_s
            layout{ov::PartialShape{e, 32, gate_groups, 1}, data_types::u8, format::bfyx},    // gate_z
            layout{ov::PartialShape{e, 32, gate_groups, 1}, data_types::u8, format::bfyx},    // up_z
            layout{ov::PartialShape{e, 64, down_groups, 1}, data_types::u8, format::bfyx}};   // down_z

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

    ~OtdPackLayerFixture() {
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

    moe_otd::PackLayout open_pack() const {
        auto pack = moe_otd::PackLayout::open(pack_path(), spec());
        OPENVINO_ASSERT(pack.has_value(), "fixture pack did not open");
        return *pack;
    }

    void build_pack() const {
        auto provider = make_provider();
        provider->configure_pack(moe_otd::PackMode::build, pack_dir, layouts);
        OPENVINO_ASSERT(provider->uses_pack(), "fixture pack was not built");
    }

    void flip_record_byte(uint32_t expert, size_t tensor, uint64_t byte) const {
        const auto pack = open_pack();
        const auto offset = pack.record_offset(expert) + pack.tensor_offset_in_record(tensor) + byte;
        std::fstream io(pack_path(), std::ios::binary | std::ios::in | std::ios::out);
        io.seekg(static_cast<std::streamoff>(offset));
        char c = 0;
        io.read(&c, 1);
        c = static_cast<char>(c ^ 0x5A);
        io.seekp(static_cast<std::streamoff>(offset));
        io.write(&c, 1);
        OPENVINO_ASSERT(static_cast<bool>(io), "fixture byte flip failed");
    }

    std::vector<uint8_t> record_bytes(uint32_t expert, size_t tensor) const {
        const auto pack = open_pack();
        std::ifstream in(pack_path(), std::ios::binary);
        in.seekg(static_cast<std::streamoff>(pack.record_offset(expert) + pack.tensor_offset_in_record(tensor)));
        std::vector<uint8_t> data(pack.tensor_bytes(tensor));
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        return data;
    }

    // Device buffers with the full logical layouts, filled with `sentinel`; order matches `layouts`.
    std::array<memory::ptr, moe_otd::pack_tensor_count> allocate_resident(cldnn::moe_weights& weights) const {
        std::array<memory::ptr, moe_otd::pack_tensor_count> mems;
        for (size_t t = 0; t < mems.size(); t++) {
            mems[t] = get_test_engine().allocate_memory(layouts[t]);
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

}  // namespace

TEST(moe_expert_weight_provider, resident_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_FALSE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 0U);

    // Fully resident: try_acquire_simultaneous returns the expert ids unchanged, including duplicates
    // (no remap, no dedup) - the expert id is already the addressable slot.
    std::vector<uint32_t> experts = {3, 1, 4, 1, 5, 9, 2, 6};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 1, 4, 1, 5, 9, 2, 6}));
}

TEST(moe_expert_weight_provider, offload_construction_state) {
    MOECompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};
    OffloadExpertWeightProvider provider(/*capacity=*/16, config, offsets, "/path/to/weights.bin");

    ASSERT_TRUE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 16U);
    // Resident buffers are bound lazily at first execute, not at construction.
    ASSERT_FALSE(provider.is_bound());
    ASSERT_FALSE(provider.cache().is_initialized());
}

TEST(moe_expert_weight_provider, offload_bind_resident_buffers) {
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/8, config, {}, "/path/to/weights.bin");
    ASSERT_FALSE(provider.is_bound());

    cldnn::moe_weights resident;  // null device pointers are fine; only the binding flag matters here
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
    ASSERT_TRUE(provider.cache().is_initialized());

    // Idempotent: a second bind keeps the provider bound.
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
}

TEST(moe_expert_weight_provider, resident_try_acquire_simultaneous_is_identity) {
    ResidentExpertWeightProvider provider;
    std::vector<uint32_t> experts = {3, 7, 3, 1, 7};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 7, 3, 1, 7}));
}

TEST(moe_expert_weight_provider, resident_acquire_one_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_EQ(provider.acquire_one(42, get_test_stream()), 42U);
    ASSERT_EQ(provider.acquire_one(0, get_test_stream()), static_cast<size_t>(0));
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_capacity_overflow) {
    // Capacity=2 means at most 2 unique experts can be acquired simultaneously.
    // Requesting 3 unique experts must return nullopt (capacity overflow).
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate the LRU cache so experts 0,1 are hits (no disk I/O needed)
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 3 unique experts (0, 1, 2) exceeds capacity=2 -> nullopt
    std::vector<uint32_t> experts = {0, 1, 2, 0, 1};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_FALSE(lease.has_value());
}

TEST(moe_expert_weight_provider, offload_overflow_leaves_cache_untouched) {
    // An oversized lease must be rejected before any expert is streamed or evicted.
    // Loading 2 or 3 here would read "/nonexistent" and throw.
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    auto& cache = provider.cache();
    auto [slot0, miss0] = cache.get_lru_item(0);
    cache.set_filled(slot0);
    auto [slot1, miss1] = cache.get_lru_item(1);
    cache.set_filled(slot1);

    std::vector<uint32_t> experts = {2, 3, 0};
    std::optional<ov::intel_gpu::ocl::moe::ExpertSlotLease> lease;
    ASSERT_NO_THROW(lease = provider.try_acquire_simultaneous(experts, get_test_stream()));
    ASSERT_FALSE(lease.has_value());

    ASSERT_EQ(cache.evictions(), 0U);
    ASSERT_EQ(cache.size(), 2U);
    auto r0 = cache.lookup(0);
    auto r1 = cache.lookup(1);
    ASSERT_TRUE(r0.hit);
    ASSERT_TRUE(r1.hit);
    ASSERT_EQ(r0.slot, slot0);
    ASSERT_EQ(r1.slot, slot1);
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_within_capacity) {
    // Same setup but with only 2 unique experts -> should succeed
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 2 unique experts (0, 1) with duplicates -> within capacity=2
    std::vector<uint32_t> experts = {0, 1, 0, 1, 0};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    // Slots should be valid (cache returned slots for experts 0 and 1)
    ASSERT_EQ(lease->size(), 5U);
    // Duplicates must map to the same slot
    ASSERT_EQ((*lease)[0], (*lease)[2]);
    ASSERT_EQ((*lease)[0], (*lease)[4]);
    ASSERT_EQ((*lease)[1], (*lease)[3]);
}

TEST(moe_expert_weight_provider, offload_acquire_one_basic) {
    // acquire_one always succeeds (evicts LRU if needed)
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    auto [slot0, hit0] = cache.get_lru_item(5);
    cache.set_filled(slot0);

    // acquire_one should return a valid slot
    auto slot = provider.acquire_one(5, get_test_stream());
    ASSERT_EQ(slot, slot0);
}

// The tests below drive the provider's real serving path (configure_pack -> acquire_one ->
// fill_weights_from_pack) rather than the standalone verify_pack() utility.

TEST(moe_expert_weight_provider, pack_require_fails_closed_on_unusable_pack_and_use_falls_back) {
    OtdPackLayerFixture fx;

    auto missing_require = fx.make_provider();
    ASSERT_THROW(missing_require->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts), ov::Exception);
    ASSERT_FALSE(missing_require->uses_pack());

    auto missing_use = fx.make_provider();
    ASSERT_NO_THROW(missing_use->configure_pack(moe_otd::PackMode::use, fx.pack_dir, fx.layouts));
    ASSERT_FALSE(missing_use->uses_pack());

    fx.build_pack();
    auto valid_require = fx.make_provider();
    ASSERT_NO_THROW(valid_require->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts));
    ASSERT_TRUE(valid_require->uses_pack());
    valid_require.reset();

    {
        std::fstream io(fx.pack_path(), std::ios::binary | std::ios::in | std::ios::out);
        io.seekp(static_cast<std::streamoff>(offsetof(moe_otd::PackHeader, records_offset)));
        const char garbage = 0x11;
        io.write(&garbage, 1);
    }
    auto corrupt_header_require = fx.make_provider();
    ASSERT_THROW(corrupt_header_require->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts), ov::Exception);
    ASSERT_FALSE(corrupt_header_require->uses_pack());

    auto wrong_dir_require = fx.make_provider();
    ASSERT_THROW(wrong_dir_require->configure_pack(moe_otd::PackMode::require, fx.root / "absent", fx.layouts), ov::Exception);
}

TEST(moe_expert_weight_provider, pack_require_serves_verified_record_into_slot) {
    OtdPackLayerFixture fx;
    fx.build_pack();

    auto provider = fx.make_provider();
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    ASSERT_TRUE(provider->uses_pack());
    cldnn::moe_weights resident;
    const auto mems = fx.allocate_resident(resident);
    provider->bind(resident);

    const uint32_t expert = 2;
    const size_t slot = provider->acquire_one(expert, get_test_stream());
    for (size_t t = 0; t < mems.size(); t++) {
        const auto expected = fx.record_bytes(expert, t);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock(mems[t], get_test_stream());
        for (size_t i = 0; i < lock.size(); i++) {
            const bool in_slot = i >= slot * expected.size() && i < (slot + 1) * expected.size();
            const uint8_t want = in_slot ? expected[i - slot * expected.size()] : OtdPackLayerFixture::sentinel;
            ASSERT_EQ(lock[i], want) << "tensor " << t << " byte " << i;
        }
    }
    ASSERT_TRUE(provider->cache().lookup(expert).hit);
}

TEST(moe_expert_weight_provider, pack_require_rejects_corrupt_record_without_touching_device) {
    OtdPackLayerFixture fx;
    fx.build_pack();
    // A byte in the last tensor of the record: a late-copy check would already have written eight tensors.
    fx.flip_record_byte(/*expert=*/1, /*tensor=*/8, /*byte=*/3);

    auto provider = fx.make_provider();
    provider->configure_pack(moe_otd::PackMode::require, fx.pack_dir, fx.layouts);
    ASSERT_TRUE(provider->uses_pack());  // Header metadata is intact; only the record is bad.
    cldnn::moe_weights resident;
    const auto mems = fx.allocate_resident(resident);
    provider->bind(resident);

    ASSERT_THROW(provider->acquire_one(1, get_test_stream()), ov::Exception);
    for (const auto& mem : mems) {
        ASSERT_TRUE(OtdPackLayerFixture::all_sentinel(mem));
    }
    // The reserved slot is not marked filled, so the expert is never served from it as a hit.
    ASSERT_FALSE(provider->cache().lookup(1).hit);
    ASSERT_THROW(provider->acquire_one(1, get_test_stream()), ov::Exception);
    for (const auto& mem : mems) {
        ASSERT_TRUE(OtdPackLayerFixture::all_sentinel(mem));
    }

    // Other experts of the same pack still serve normally.
    ASSERT_NO_THROW(provider->acquire_one(0, get_test_stream()));
    ASSERT_TRUE(provider->cache().lookup(0).hit);
}

TEST(moe_expert_weight_provider, pack_use_mode_does_not_silently_serve_corrupt_open_pack) {
    // `use` only falls back when the pack cannot be opened at configure time. Once a pack is
    // open, a corrupt record is an error in every mode rather than a silent source switch.
    OtdPackLayerFixture fx;
    fx.build_pack();
    fx.flip_record_byte(/*expert=*/3, /*tensor=*/0, /*byte=*/0);

    auto provider = fx.make_provider();
    provider->configure_pack(moe_otd::PackMode::use, fx.pack_dir, fx.layouts);
    ASSERT_TRUE(provider->uses_pack());
    cldnn::moe_weights resident;
    const auto mems = fx.allocate_resident(resident);
    provider->bind(resident);

    ASSERT_THROW(provider->acquire_one(3, get_test_stream()), ov::Exception);
    for (const auto& mem : mems) {
        ASSERT_TRUE(OtdPackLayerFixture::all_sentinel(mem));
    }
}
