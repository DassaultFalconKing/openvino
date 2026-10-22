// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Pre-packed routed-expert files for MoE offload-to-disk (OTD).
//
// One pack file holds every routed expert of one MoE layer. Each expert is a single
// contiguous, 4 KiB-aligned record containing its nine tensors (gate/up/down weight,
// scale, zero-point) already in the device layout the kernels consume, so an OTD miss is
// one positional read with no scale/zero-point transpose. Records are addressed through a
// placement table, which lets a re-pack put co-activated experts next to each other
// without changing readers.
//
// The file name is a content key over the source weights identity, tensor offsets and
// sizes, layer configuration and the pack format version, so a different model, export
// or quantization never matches a stale pack.
namespace ov::intel_gpu::ocl::moe_otd {

constexpr size_t pack_tensor_count = 9;
constexpr uint32_t pack_format_version = 1;
constexpr size_t pack_alignment = 4096;

enum class PackMode { off, use, require, build };

std::optional<PackMode> parse_pack_mode(std::string_view text);

// Everything that determines the bytes of a layer pack.
struct PackLayerSpec {
    uint64_t source_file_size = 0;
    // Fingerprint of sampled source bytes (see sample_ranges()).
    uint64_t source_sample_hash = 0;
    std::array<uint64_t, pack_tensor_count> source_offsets{};
    // Per-expert bytes of each tensor; 0 marks an absent tensor.
    std::array<uint64_t, pack_tensor_count> tensor_bytes{};
    uint64_t num_expert = 0;
    // Layer configuration and element types that influence the device layout transform.
    std::vector<uint64_t> layout_params;

    uint64_t key() const;
    std::string file_name() const;
};

struct ByteRange {
    uint64_t offset = 0;
    uint64_t size = 0;
};

// Source ranges hashed into source_sample_hash: the head of expert 0 and of the last expert for each tensor.
std::vector<ByteRange> sample_ranges(const PackLayerSpec& spec, uint64_t max_bytes_per_range = 64 * 1024);

#pragma pack(push, 1)
struct PackHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t layer_key;
    uint64_t num_expert;
    uint64_t record_stride;
    uint64_t records_offset;
    uint64_t placement_offset;
    uint64_t checksums_offset;
    uint64_t tensor_bytes[pack_tensor_count];
    uint64_t tensor_offset_in_record[pack_tensor_count];
    uint64_t header_checksum;
};
#pragma pack(pop)

// Validated, in-memory view of a pack header plus its placement table.
class PackLayout {
public:
    static PackLayout plan(const PackLayerSpec& spec);

    uint64_t key() const {
        return _header.layer_key;
    }
    uint64_t num_expert() const {
        return _header.num_expert;
    }
    uint64_t record_stride() const {
        return _header.record_stride;
    }
    // Bytes of a record that carry tensor data (excludes alignment padding).
    uint64_t record_payload_bytes() const;
    uint64_t tensor_bytes(size_t tensor) const {
        return _header.tensor_bytes[tensor];
    }
    uint64_t tensor_offset_in_record(size_t tensor) const {
        return _header.tensor_offset_in_record[tensor];
    }
    uint64_t record_offset(uint64_t expert) const;
    uint64_t file_size() const;
    const PackHeader& header() const {
        return _header;
    }
    const std::vector<uint32_t>& placement() const {
        return _placement;
    }

    // Checks exactly the payload bytes of a logical expert. Checksums in pack v1 are
    // indexed by expert ID, NOT by its physical placement index.
    bool verify_record_payload(uint32_t expert, const void* data, size_t bytes) const;

    // Replaces the expert -> record-position mapping; must be a permutation of [0, num_expert).
    void set_placement(std::vector<uint32_t> placement);

    // Reads and validates the header and placement of an existing pack against the expected spec.
    // Returns nullopt with a reason when the file is missing, truncated, corrupt or for another spec.
    static std::optional<PackLayout> open(const std::filesystem::path& path, const PackLayerSpec& expected, std::string* reason = nullptr);

private:
    void finalize_header();

    PackHeader _header{};
    std::vector<uint32_t> _placement;
    std::vector<uint64_t> _checksums;
};

// A record whose bytes do not match the checksum stored for its logical expert. Distinct from
// I/O errors so callers can count and report integrity rejections separately.
class PackIntegrityError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The serving read boundary: one positional read, then checksum the staging bytes
// before any caller is allowed to copy them to a resident GPU slot. Reader must
// throw on short reads. Kept header-only to preserve a direct, allocation-free
// invocation with ParallelWeightReader and with an offline test reader.
// Throws PackIntegrityError on mismatch; `staging` then holds rejected bytes that must not be used.
template <typename Reader>
void read_verified_record(const PackLayout& pack, uint32_t expert, std::vector<uint8_t>& staging, Reader&& reader) {
    const auto payload = pack.record_payload_bytes();
    if (payload > staging.max_size()) {
        throw std::length_error("OTD pack: expert record exceeds staging capacity");
    }
    staging.resize(static_cast<size_t>(payload));
    const auto offset = pack.record_offset(expert);
    reader(reinterpret_cast<char*>(staging.data()), staging.size(), offset);
    if (!pack.verify_record_payload(expert, staging.data(), staging.size())) {
        throw PackIntegrityError("OTD pack: checksum mismatch for expert " + std::to_string(expert) + " (record position " +
                                 std::to_string(pack.placement()[expert]) + ", " + std::to_string(staging.size()) + " payload bytes)");
    }
}

// Supplies the device-layout bytes of one tensor of one expert.
using PackTensorSource = std::function<void(uint32_t expert, size_t tensor, std::vector<uint8_t>& out)>;

struct PackBuildResult {
    std::filesystem::path path;
    uint64_t bytes_written = 0;
    bool reused_existing = false;
};

// Writes the pack for `spec` into `dir` unless a valid one is already present. The file is
// written under a temporary name and renamed into place, so readers never see a partial pack.
PackBuildResult build_pack(const std::filesystem::path& dir,
                           const PackLayerSpec& spec,
                           const PackTensorSource& source,
                           const std::vector<uint32_t>& placement = {});

// Re-reads every record and compares it with the stored checksum. Returns the number of corrupt records.
uint64_t verify_pack(const std::filesystem::path& path, const PackLayerSpec& spec, std::string* reason = nullptr);

}  // namespace ov::intel_gpu::ocl::moe_otd
