// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "ocl_v2/moe/moe_otd_pack.hpp"
#include "test_utils/test_utils.h"

using namespace ov::intel_gpu::ocl::moe_otd;

namespace {

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        path = std::filesystem::temp_directory_path() / ("moe_otd_pack_test_" + std::to_string(rd()) + std::to_string(rd()));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    std::filesystem::path path;
};

PackLayerSpec make_spec(uint64_t num_expert = 6) {
    PackLayerSpec spec;
    spec.num_expert = num_expert;
    spec.source_file_size = 1 << 20;
    spec.source_sample_hash = 0x1234;
    // Deliberately unaligned sizes; zero-point tensors absent for down.
    spec.tensor_bytes = {1000, 1000, 777, 40, 40, 31, 20, 20, 0};
    for (size_t t = 0; t < pack_tensor_count; t++) {
        spec.source_offsets[t] = 4096 * (t + 1);
    }
    spec.layout_params = {2048, 512, 128, 1};
    return spec;
}

uint8_t byte_of(uint32_t expert, size_t tensor, size_t i) {
    return static_cast<uint8_t>((expert * 131 + tensor * 17 + i * 7) & 0xFF);
}

PackTensorSource pattern_source(const PackLayerSpec& spec, size_t* calls = nullptr) {
    return [spec, calls](uint32_t expert, size_t tensor, std::vector<uint8_t>& out) {
        if (calls) {
            (*calls)++;
        }
        out.resize(spec.tensor_bytes[tensor]);
        for (size_t i = 0; i < out.size(); i++) {
            out[i] = byte_of(expert, tensor, i);
        }
    };
}

std::vector<uint8_t> read_at(const std::filesystem::path& path, uint64_t offset, uint64_t size) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(offset));
    std::vector<uint8_t> data(size);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    return data;
}

void flip_byte(const std::filesystem::path& path, uint64_t offset) {
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    io.seekg(static_cast<std::streamoff>(offset));
    char c = 0;
    io.read(&c, 1);
    c = static_cast<char>(c ^ 0x5A);
    io.seekp(static_cast<std::streamoff>(offset));
    io.write(&c, 1);
}


// Exercise the same read-then-check boundary used by fill_weights_from_pack.
// This is deliberately independent of the whole-file verify_pack() utility.
std::vector<uint8_t> serving_read(const PackLayout& layout,
                                  const std::filesystem::path& path,
                                  uint32_t expert,
                                  size_t& reads) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("OTD test reader: missing pack");
    }
    std::vector<uint8_t> staging;
    read_verified_record(layout, expert, staging, [&](char* dst, size_t size, uint64_t offset) {
        ++reads;
        in.clear();
        in.seekg(static_cast<std::streamoff>(offset));
        if (!in.read(dst, static_cast<std::streamsize>(size))) {
            throw std::runtime_error("OTD test reader: short read");
        }
    });
    return staging;
}

uint64_t resealed_header_hash(PackHeader header) {
    header.header_checksum = 0;
    uint64_t hash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&header);
    for (size_t i = 0; i < sizeof(PackHeader); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void write_header(const std::filesystem::path& path, PackHeader header) {
    header.header_checksum = resealed_header_hash(header);
    std::fstream out(path, std::ios::in | std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!out) {
        throw std::runtime_error("OTD test writer: header update failed");
    }
}

}  // namespace

TEST(moe_otd_pack, parse_mode) {
    ASSERT_EQ(parse_pack_mode(""), PackMode::off);
    ASSERT_EQ(parse_pack_mode("off"), PackMode::off);
    ASSERT_EQ(parse_pack_mode("use"), PackMode::use);
    ASSERT_EQ(parse_pack_mode("require"), PackMode::require);
    ASSERT_EQ(parse_pack_mode("build"), PackMode::build);
    ASSERT_FALSE(parse_pack_mode("BUILD").has_value());
}

TEST(moe_otd_pack, plan_is_aligned_and_covers_all_tensors) {
    const auto spec = make_spec();
    const auto layout = PackLayout::plan(spec);
    ASSERT_EQ(layout.record_stride() % pack_alignment, 0U);
    ASSERT_EQ(layout.header().records_offset % pack_alignment, 0U);
    ASSERT_GE(layout.record_stride(), layout.record_payload_bytes());
    uint64_t prev_end = 0;
    for (size_t t = 0; t < pack_tensor_count; t++) {
        ASSERT_EQ(layout.tensor_offset_in_record(t) % 64, 0U);
        ASSERT_GE(layout.tensor_offset_in_record(t), prev_end);
        prev_end = layout.tensor_offset_in_record(t) + layout.tensor_bytes(t);
    }
    ASSERT_EQ(layout.record_offset(0), layout.header().records_offset);
    ASSERT_EQ(layout.record_offset(5), layout.header().records_offset + 5 * layout.record_stride());
    ASSERT_EQ(layout.file_size(), layout.header().records_offset + 6 * layout.record_stride());
}

TEST(moe_otd_pack, key_depends_on_every_identity_field) {
    const auto base = make_spec();
    auto changed = [&](auto mutate) {
        auto s = base;
        mutate(s);
        return s.key() != base.key() && s.file_name() != base.file_name();
    };
    ASSERT_TRUE(changed([](PackLayerSpec& s) {
        s.source_file_size++;
    }));
    ASSERT_TRUE(changed([](PackLayerSpec& s) {
        s.source_sample_hash++;
    }));
    ASSERT_TRUE(changed([](PackLayerSpec& s) {
        s.source_offsets[4]++;
    }));
    ASSERT_TRUE(changed([](PackLayerSpec& s) {
        s.tensor_bytes[0]++;
    }));
    ASSERT_TRUE(changed([](PackLayerSpec& s) {
        s.num_expert++;
    }));
    ASSERT_TRUE(changed([](PackLayerSpec& s) {
        s.layout_params[2] = 64;
    }));
    ASSERT_EQ(base.file_name().rfind("otd-", 0), 0U);
}

TEST(moe_otd_pack, sample_ranges_cover_first_and_last_expert) {
    const auto spec = make_spec();
    const auto ranges = sample_ranges(spec, 100);
    // Eight present tensors, two ranges each.
    ASSERT_EQ(ranges.size(), 16U);
    ASSERT_EQ(ranges[0].offset, spec.source_offsets[0]);
    ASSERT_EQ(ranges[0].size, 100U);
    ASSERT_EQ(ranges[1].offset, spec.source_offsets[0] + 5 * spec.tensor_bytes[0]);
    ASSERT_EQ(ranges[6].size, 40U);
}

TEST(moe_otd_pack, build_open_and_records_round_trip) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    ASSERT_FALSE(built.reused_existing);
    ASSERT_EQ(built.path, dir.path / spec.file_name());

    std::string reason;
    auto layout = PackLayout::open(built.path, spec, &reason);
    ASSERT_TRUE(layout.has_value()) << reason;
    ASSERT_EQ(std::filesystem::file_size(built.path), layout->file_size());

    for (uint32_t e = 0; e < spec.num_expert; e++) {
        const auto record = read_at(built.path, layout->record_offset(e), layout->record_payload_bytes());
        for (size_t t = 0; t < pack_tensor_count; t++) {
            for (size_t i = 0; i < spec.tensor_bytes[t]; i++) {
                ASSERT_EQ(record[layout->tensor_offset_in_record(t) + i], byte_of(e, t, i)) << "expert " << e << " tensor " << t << " byte " << i;
            }
        }
    }
    ASSERT_EQ(verify_pack(built.path, spec, &reason), 0U) << reason;

    // No temporary files are left behind.
    size_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir.path)) {
        (void)entry;
        files++;
    }
    ASSERT_EQ(files, 1U);
}

TEST(moe_otd_pack, existing_valid_pack_is_reused_without_reading_source) {
    TempDir dir;
    const auto spec = make_spec();
    build_pack(dir.path, spec, pattern_source(spec));
    size_t calls = 0;
    const auto again = build_pack(dir.path, spec, pattern_source(spec, &calls));
    ASSERT_TRUE(again.reused_existing);
    ASSERT_EQ(calls, 0U);
}

TEST(moe_otd_pack, pack_for_other_spec_is_rejected) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    auto other = spec;
    other.source_sample_hash++;
    std::string reason;
    ASSERT_FALSE(PackLayout::open(built.path, other, &reason).has_value());
    ASSERT_FALSE(reason.empty());
    ASSERT_FALSE(PackLayout::open(dir.path / "absent.pack", spec, &reason).has_value());
    ASSERT_EQ(reason, "missing");
}

TEST(moe_otd_pack, corrupt_record_is_detected_and_not_reused) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    auto layout = PackLayout::open(built.path, spec);
    ASSERT_TRUE(layout.has_value());
    flip_byte(built.path, layout->record_offset(3) + layout->tensor_offset_in_record(2) + 5);
    ASSERT_EQ(verify_pack(built.path, spec), 1U);

    // Header is intact, so only the full record check can catch this; build must not reuse it.
    ASSERT_TRUE(PackLayout::open(built.path, spec).has_value());
    const auto rebuilt = build_pack(dir.path, spec, pattern_source(spec));
    ASSERT_FALSE(rebuilt.reused_existing);
    ASSERT_EQ(verify_pack(built.path, spec), 0U);
}

TEST(moe_otd_pack, corrupt_header_or_truncation_fails_open_and_triggers_rebuild) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    flip_byte(built.path, offsetof(PackHeader, record_stride));
    std::string reason;
    ASSERT_FALSE(PackLayout::open(built.path, spec, &reason).has_value());
    ASSERT_EQ(reason, "header checksum mismatch");

    size_t calls = 0;
    const auto rebuilt = build_pack(dir.path, spec, pattern_source(spec, &calls));
    ASSERT_FALSE(rebuilt.reused_existing);
    ASSERT_GT(calls, 0U);
    ASSERT_TRUE(PackLayout::open(built.path, spec).has_value());

    std::filesystem::resize_file(built.path, std::filesystem::file_size(built.path) - 1);
    ASSERT_FALSE(PackLayout::open(built.path, spec, &reason).has_value());
    ASSERT_EQ(reason, "truncated records");
}

TEST(moe_otd_pack, placement_reorders_records_without_changing_contents) {
    TempDir dir;
    const auto spec = make_spec();
    const std::vector<uint32_t> placement = {5, 0, 4, 1, 3, 2};
    const auto built = build_pack(dir.path, spec, pattern_source(spec), placement);
    auto layout = PackLayout::open(built.path, spec);
    ASSERT_TRUE(layout.has_value());
    ASSERT_EQ(layout->placement(), placement);
    ASSERT_EQ(layout->record_offset(0), layout->header().records_offset + 5 * layout->record_stride());
    for (uint32_t e = 0; e < spec.num_expert; e++) {
        const auto record = read_at(built.path, layout->record_offset(e), spec.tensor_bytes[0]);
        ASSERT_EQ(record[0], byte_of(e, 0, 0));
        ASSERT_EQ(record[999], byte_of(e, 0, 999));
    }
    ASSERT_EQ(verify_pack(built.path, spec), 0U);
    ASSERT_THROW(PackLayout::plan(spec).set_placement({0, 0, 1, 2, 3, 4}), std::invalid_argument);
}


TEST(moe_otd_pack, serving_read_validates_logical_expert_before_gpu_copy) {
    TempDir dir;
    const auto spec = make_spec();
    const std::vector<uint32_t> placement = {5, 0, 4, 1, 3, 2};
    const auto built = build_pack(dir.path, spec, pattern_source(spec), placement);
    const auto pack = PackLayout::open(built.path, spec);
    ASSERT_TRUE(pack.has_value());

    size_t reads = 0;
    for (uint32_t e = 0; e < spec.num_expert; ++e) {
        const auto record = serving_read(*pack, built.path, e, reads);
        ASSERT_EQ(record[0], byte_of(e, 0, 0));
    }
    ASSERT_EQ(reads, spec.num_expert);
    ASSERT_FALSE(pack->verify_record_payload(static_cast<uint32_t>(spec.num_expert), nullptr, 0));

    // Detect corruption of first and last logical experts, including one
    // physically stored at the opposite end of the placement permutation.
    for (const uint32_t e : {uint32_t{0}, static_cast<uint32_t>(spec.num_expert - 1)}) {
        const auto offset = pack->record_offset(e) + pack->tensor_offset_in_record(0);
        flip_byte(built.path, offset);
        const auto before = reads;
        EXPECT_THROW(serving_read(*pack, built.path, e, reads), std::runtime_error);
        EXPECT_EQ(reads, before + 1);
        flip_byte(built.path, offset);
    }
    ASSERT_EQ(verify_pack(built.path, spec), 0U);
}

TEST(moe_otd_pack, serving_read_rejects_corrupt_checksum_and_truncated_record) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    const auto pack = PackLayout::open(built.path, spec);
    ASSERT_TRUE(pack.has_value());

    // Checksum table is keyed by logical expert, not record position.
    flip_byte(built.path, pack->header().checksums_offset + 2 * sizeof(uint64_t));
    const auto changed_table = PackLayout::open(built.path, spec);
    ASSERT_TRUE(changed_table.has_value());
    size_t reads = 0;
    EXPECT_THROW(serving_read(*changed_table, built.path, 2, reads), std::runtime_error);
    EXPECT_EQ(reads, 1U);
    EXPECT_NO_THROW(serving_read(*changed_table, built.path, 3, reads));

    // Metadata was read before truncation: even a late, short runtime read
    // must fail closed rather than return zero-filled or stale data.
    std::filesystem::resize_file(built.path, pack->record_offset(5) + 8);
    EXPECT_THROW(serving_read(*pack, built.path, 5, reads), std::runtime_error);
    std::string reason;
    EXPECT_FALSE(PackLayout::open(built.path, spec, &reason));
    EXPECT_EQ(reason, "truncated records");
}

TEST(moe_otd_pack, resealed_noncanonical_offsets_and_sizes_are_rejected) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    const auto pack = PackLayout::open(built.path, spec);
    ASSERT_TRUE(pack.has_value());
    const PackHeader good = pack->header();

    const std::vector<std::function<void(PackHeader&)>> mutations = {
        [](PackHeader& h) { h.header_bytes = 0; },
        [](PackHeader& h) { h.placement_offset = 1; },
        [](PackHeader& h) { h.checksums_offset = 1; },
        [](PackHeader& h) { h.records_offset = std::numeric_limits<uint64_t>::max() - 1; },
        [](PackHeader& h) { h.record_stride = std::numeric_limits<uint64_t>::max(); },
        [](PackHeader& h) { h.tensor_offset_in_record[2] = std::numeric_limits<uint64_t>::max(); },
    };
    for (const auto& mutate : mutations) {
        auto forged = good;
        mutate(forged);
        write_header(built.path, forged);
        std::string reason;
        EXPECT_FALSE(PackLayout::open(built.path, spec, &reason)) << reason;
        EXPECT_EQ(reason, "non-canonical pack header layout");
    }
    write_header(built.path, good);
    EXPECT_TRUE(PackLayout::open(built.path, spec));
}

TEST(moe_otd_pack, invalid_placement_table_and_overflowed_plan_fail_closed) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    const auto pack = PackLayout::open(built.path, spec);
    ASSERT_TRUE(pack.has_value());
    flip_byte(built.path, pack->header().placement_offset);
    std::string reason;
    EXPECT_FALSE(PackLayout::open(built.path, spec, &reason));
    EXPECT_EQ(reason, "corrupt placement table");

    auto overflow = spec;
    overflow.tensor_bytes[0] = std::numeric_limits<uint64_t>::max();
    EXPECT_THROW(PackLayout::plan(overflow), std::overflow_error);
    auto zero_experts = spec;
    zero_experts.num_expert = 0;
    EXPECT_THROW(PackLayout::plan(zero_experts), std::invalid_argument);
    EXPECT_NO_THROW(PackLayout::plan(make_spec(1)));
}

TEST(moe_otd_pack, failing_source_leaves_no_pack) {
    TempDir dir;
    const auto spec = make_spec();
    auto failing = [spec](uint32_t expert, size_t tensor, std::vector<uint8_t>& out) {
        if (expert == 2) {
            throw std::runtime_error("read error");
        }
        out.resize(spec.tensor_bytes[tensor]);
    };
    ASSERT_THROW(build_pack(dir.path, spec, failing), std::runtime_error);
    ASSERT_TRUE(std::filesystem::is_empty(dir.path));

    auto short_source = [](uint32_t, size_t, std::vector<uint8_t>& out) {
        out.resize(1);
    };
    ASSERT_THROW(build_pack(dir.path, spec, short_source), std::runtime_error);
    ASSERT_TRUE(std::filesystem::is_empty(dir.path));
}
