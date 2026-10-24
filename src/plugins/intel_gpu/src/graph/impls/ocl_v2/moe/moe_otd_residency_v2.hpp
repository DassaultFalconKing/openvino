// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Offline-scored, resident-priority tiers for GPU MoE expert loading.
// This is not a new allocator: all slots remain GPU USM device allocations.
// Hot/warm/cold are SOFT eviction priorities; demand always makes progress.
// Arm/prefetch requires a separate GPU reader-lifetime and memory-admission gate.
#include "moe_otd_placement_v2.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::intel_gpu::ocl::moe_otd {

inline std::vector<uint8_t> parse_residency_v2(const std::vector<uint8_t>& bytes,
                                                const PackLayerSpec& spec,
                                                size_t resident_capacity) {
    static constexpr std::array<uint8_t, 8> magic = {'O','T','D','R','E','S','I','D'};
    if (spec.num_expert == 0 || spec.num_expert > 16384 || bytes.size() != 32 + spec.num_expert) {
        throw std::runtime_error("OTD residency: invalid byte length/count");
    }
    if (!std::equal(magic.begin(), magic.end(), bytes.begin()) ||
        placement_le32(bytes.data() + 8) != 2 ||
        placement_le32(bytes.data() + 12) != spec.num_expert ||
        placement_le64(bytes.data() + 16) != spec.key()) {
        throw std::runtime_error("OTD residency: wrong magic, version, count or compiled pack key");
    }
    if (placement_le64(bytes.data() + bytes.size() - 8) !=
        placement_fnv64(std::vector<uint8_t>(bytes.begin(), bytes.end() - 8))) {
        throw std::runtime_error("OTD residency: checksum mismatch");
    }
    std::vector<uint8_t> tiers(bytes.begin() + 24, bytes.end() - 8);
    // Profile metadata cannot reserve more slots than the compiled GPU arena.
    // Retain one demand slot when capacity > 1; downgrade excess hot experts
    // rather than starving unobserved or newly hot requests.
    const size_t hot_limit = resident_capacity > 1 ? resident_capacity - 1 : 0;
    size_t hot = 0;
    for (uint8_t& tier : tiers) {
        if (tier > 2) throw std::runtime_error("OTD residency: invalid tier");
        if (tier == 2 && ++hot > hot_limit) tier = 1;
    }
    return tiers;
}

inline std::vector<uint8_t> load_residency_v2(const std::filesystem::path& dir,
                                               const PackLayerSpec& spec,
                                               size_t resident_capacity) {
    const auto root = std::filesystem::weakly_canonical(dir);
    if (!std::filesystem::is_directory(root)) {
        throw std::runtime_error("OTD residency: directory missing");
    }
    auto name = spec.file_name();
    name.replace(name.size() - 5, 5, ".residency");
    const auto path = root / name;
    if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("OTD residency: missing/unsafe priority file");
    }
    const auto size = std::filesystem::file_size(path);
    if (spec.num_expert == 0 || spec.num_expert > 16384 || size != 32 + spec.num_expert) {
        throw std::runtime_error("OTD residency: invalid priority file length");
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("OTD residency: unable to open priorities");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file || file.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("OTD residency: short read");
    }
    return parse_residency_v2(bytes, spec, resident_capacity);
}

} // namespace ov::intel_gpu::ocl::moe_otd
