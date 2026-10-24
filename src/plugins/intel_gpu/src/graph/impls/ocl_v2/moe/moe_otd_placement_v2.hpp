// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Profile-guided placement handoff for native V1 expert record builders.
// Deliberately header-only to avoid modifying the GPU build-system file list.
// This is the native BUILD bridge for Pack V2 generations, not a V2 record reader.
#include "moe_otd_pack.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::intel_gpu::ocl::moe_otd {

inline uint64_t placement_fnv64(const std::vector<uint8_t>& data) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint8_t b : data) {
        hash = (hash ^ b) * UINT64_C(1099511628211);
    }
    return hash;
}

inline uint32_t placement_le32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

inline uint64_t placement_le64(const uint8_t* p) {
    uint64_t result = 0;
    for (unsigned i = 0; i < 8; ++i) result |= uint64_t(p[i]) << (8 * i);
    return result;
}

// On disk: "OTDPLACE" [8] + version [u32 LE=2] + expert_count [u32 LE]
// + native PackLayerSpec.key [u64 LE] + placement[count] [u32 LE]
// + fnv64(header+placement) [u64 LE]. All fields are exact and bounded.
// The separate V2 manifest uses SHA-256 for strong file binding.
inline std::vector<uint32_t> parse_placement_v2(const std::vector<uint8_t>& bytes,
                                                 const PackLayerSpec& spec) {
    static constexpr std::array<uint8_t, 8> magic = {'O','T','D','P','L','A','C','E'};
    if (spec.num_expert == 0 || spec.num_expert > 16384) {
        throw std::runtime_error("OTD placement: invalid compiled expert count");
    }
    const size_t expected = 32 + static_cast<size_t>(spec.num_expert) * 4;
    if (bytes.size() != expected || !std::equal(magic.begin(), magic.end(), bytes.begin())) {
        throw std::runtime_error("OTD placement: wrong magic or byte length");
    }
    if (placement_le32(bytes.data() + 8) != 2 ||
        placement_le32(bytes.data() + 12) != spec.num_expert ||
        placement_le64(bytes.data() + 16) != spec.key()) {
        throw std::runtime_error("OTD placement: wrong version, count or compiled pack key");
    }
    if (placement_le64(bytes.data() + expected - 8) !=
        placement_fnv64(std::vector<uint8_t>(bytes.begin(), bytes.end() - 8))) {
        throw std::runtime_error("OTD placement: checksum mismatch");
    }
    std::vector<uint32_t> placement(static_cast<size_t>(spec.num_expert));
    std::vector<bool> seen(static_cast<size_t>(spec.num_expert), false);
    for (size_t i = 0; i < placement.size(); ++i) {
        const uint32_t position = placement_le32(bytes.data() + 24 + i * 4);
        if (position >= spec.num_expert || seen[position]) {
            throw std::runtime_error("OTD placement: not a permutation");
        }
        seen[position] = true;
        placement[i] = position;
    }
    return placement;
}

inline std::vector<uint32_t> load_placement_v2(const std::filesystem::path& dir,
                                                const PackLayerSpec& spec) {
    // A configured placement directory means all compiled offload layers must
    // be pinned, not silently fall back to identity.
    const std::filesystem::path base = std::filesystem::weakly_canonical(dir);
    if (!std::filesystem::is_directory(base)) {
        throw std::runtime_error("OTD placement: directory not accessible");
    }
    const std::filesystem::path file =
        base / (spec.file_name().substr(0, spec.file_name().size() - 5) + ".placement");
    if (std::filesystem::is_symlink(file) || !std::filesystem::is_regular_file(file)) {
        throw std::runtime_error("OTD placement: missing or unsafe layer plan");
    }
    const auto size = std::filesystem::file_size(file);
    if (spec.num_expert == 0 || spec.num_expert > 16384 ||
        size != 32 + spec.num_expert * 4) {
        throw std::runtime_error("OTD placement: invalid file size");
    }
    std::ifstream stream(file, std::ios::binary);
    if (!stream) throw std::runtime_error("OTD placement: cannot open layer plan");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream || stream.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("OTD placement: truncated file");
    }
    // Read a stable path only during a controlled, immutable build.
    return parse_placement_v2(bytes, spec);
}

} // namespace ov::intel_gpu::ocl::moe_otd
