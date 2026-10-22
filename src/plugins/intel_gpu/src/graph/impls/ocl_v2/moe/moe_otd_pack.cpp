// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_pack.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace ov::intel_gpu::ocl::moe_otd {

namespace {

constexpr char pack_magic[8] = {'O', 'T', 'D', 'P', 'A', 'C', 'K', '1'};
constexpr uint64_t fnv_offset = 14695981039346656037ULL;
constexpr uint64_t fnv_prime = 1099511628211ULL;
constexpr uint64_t tensor_alignment = 64;

uint64_t fnv_update(uint64_t hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= fnv_prime;
    }
    return hash;
}

uint64_t fnv_u64(uint64_t hash, uint64_t value) {
    return fnv_update(hash, &value, sizeof(value));
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

uint64_t header_checksum_of(PackHeader header) {
    header.header_checksum = 0;
    return fnv_update(fnv_offset, &header, sizeof(header));
}

void set_reason(std::string* reason, const std::string& text) {
    if (reason) {
        *reason = text;
    }
}

bool is_permutation_of_range(const std::vector<uint32_t>& placement, uint64_t n) {
    if (placement.size() != n) {
        return false;
    }
    std::vector<bool> seen(n, false);
    for (uint32_t pos : placement) {
        if (pos >= n || seen[pos]) {
            return false;
        }
        seen[pos] = true;
    }
    return true;
}

}  // namespace

std::optional<PackMode> parse_pack_mode(std::string_view text) {
    if (text.empty() || text == "off") {
        return PackMode::off;
    }
    if (text == "use") {
        return PackMode::use;
    }
    if (text == "require") {
        return PackMode::require;
    }
    if (text == "build") {
        return PackMode::build;
    }
    return std::nullopt;
}

uint64_t PackLayerSpec::key() const {
    uint64_t h = fnv_update(fnv_offset, pack_magic, sizeof(pack_magic));
    h = fnv_u64(h, pack_format_version);
    h = fnv_u64(h, source_file_size);
    h = fnv_u64(h, source_sample_hash);
    for (auto v : source_offsets) {
        h = fnv_u64(h, v);
    }
    for (auto v : tensor_bytes) {
        h = fnv_u64(h, v);
    }
    h = fnv_u64(h, num_expert);
    h = fnv_u64(h, layout_params.size());
    for (auto v : layout_params) {
        h = fnv_u64(h, v);
    }
    return h;
}

std::string PackLayerSpec::file_name() const {
    std::ostringstream name;
    name << "otd-" << std::hex << std::setw(16) << std::setfill('0') << key() << ".pack";
    return name.str();
}

std::vector<ByteRange> sample_ranges(const PackLayerSpec& spec, uint64_t max_bytes_per_range) {
    std::vector<ByteRange> ranges;
    if (spec.num_expert == 0) {
        return ranges;
    }
    for (size_t t = 0; t < pack_tensor_count; t++) {
        const uint64_t size = spec.tensor_bytes[t];
        if (size == 0) {
            continue;
        }
        const uint64_t take = std::min(size, max_bytes_per_range);
        ranges.push_back({spec.source_offsets[t], take});
        if (spec.num_expert > 1) {
            ranges.push_back({spec.source_offsets[t] + (spec.num_expert - 1) * size, take});
        }
    }
    return ranges;
}

PackLayout PackLayout::plan(const PackLayerSpec& spec) {
    if (spec.num_expert == 0 || spec.num_expert > UINT32_MAX) {
        throw std::invalid_argument("OTD pack: invalid expert count");
    }
    PackLayout layout;
    auto& h = layout._header;
    std::memcpy(h.magic, pack_magic, sizeof(pack_magic));
    h.version = pack_format_version;
    h.header_bytes = static_cast<uint32_t>(pack_alignment);
    h.layer_key = spec.key();
    h.num_expert = spec.num_expert;

    uint64_t cursor = 0;
    for (size_t t = 0; t < pack_tensor_count; t++) {
        h.tensor_bytes[t] = spec.tensor_bytes[t];
        h.tensor_offset_in_record[t] = cursor;
        cursor = align_up(cursor + spec.tensor_bytes[t], tensor_alignment);
    }
    if (cursor == 0) {
        throw std::invalid_argument("OTD pack: layer has no tensor bytes");
    }
    h.record_stride = align_up(cursor, pack_alignment);
    h.placement_offset = pack_alignment;
    h.checksums_offset = h.placement_offset + align_up(spec.num_expert * sizeof(uint32_t), 8);
    h.records_offset = align_up(h.checksums_offset + spec.num_expert * sizeof(uint64_t), pack_alignment);

    layout._placement.resize(spec.num_expert);
    for (uint32_t e = 0; e < spec.num_expert; e++) {
        layout._placement[e] = e;
    }
    layout.finalize_header();
    return layout;
}

void PackLayout::finalize_header() {
    _header.header_checksum = header_checksum_of(_header);
}

uint64_t PackLayout::record_payload_bytes() const {
    uint64_t end = 0;
    for (size_t t = 0; t < pack_tensor_count; t++) {
        if (_header.tensor_bytes[t] != 0) {
            end = std::max(end, _header.tensor_offset_in_record[t] + _header.tensor_bytes[t]);
        }
    }
    return end;
}

uint64_t PackLayout::record_offset(uint64_t expert) const {
    if (expert >= _placement.size()) {
        throw std::out_of_range("OTD pack: expert out of range");
    }
    return _header.records_offset + static_cast<uint64_t>(_placement[expert]) * _header.record_stride;
}

uint64_t PackLayout::file_size() const {
    return _header.records_offset + _header.num_expert * _header.record_stride;
}

void PackLayout::set_placement(std::vector<uint32_t> placement) {
    if (!is_permutation_of_range(placement, _header.num_expert)) {
        throw std::invalid_argument("OTD pack: placement is not a permutation of the expert range");
    }
    _placement = std::move(placement);
}

std::optional<PackLayout> PackLayout::open(const std::filesystem::path& path, const PackLayerSpec& expected, std::string* reason) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        set_reason(reason, "missing");
        return std::nullopt;
    }
    const uint64_t actual_size = std::filesystem::file_size(path, ec);
    if (ec) {
        set_reason(reason, "cannot stat");
        return std::nullopt;
    }
    std::ifstream in(path, std::ios::binary);
    PackLayout layout;
    if (!in.read(reinterpret_cast<char*>(&layout._header), sizeof(PackHeader))) {
        set_reason(reason, "truncated header");
        return std::nullopt;
    }
    const auto& h = layout._header;
    if (std::memcmp(h.magic, pack_magic, sizeof(pack_magic)) != 0 || h.version != pack_format_version) {
        set_reason(reason, "not an OTD pack of this version");
        return std::nullopt;
    }
    if (h.header_checksum != header_checksum_of(h)) {
        set_reason(reason, "header checksum mismatch");
        return std::nullopt;
    }
    if (h.layer_key != expected.key() || h.num_expert != expected.num_expert) {
        set_reason(reason, "pack belongs to a different model or layer");
        return std::nullopt;
    }
    for (size_t t = 0; t < pack_tensor_count; t++) {
        if (h.tensor_bytes[t] != expected.tensor_bytes[t]) {
            set_reason(reason, "tensor size mismatch");
            return std::nullopt;
        }
    }
    if (actual_size < layout.file_size()) {
        set_reason(reason, "truncated records");
        return std::nullopt;
    }
    std::vector<uint32_t> placement(h.num_expert);
    in.seekg(static_cast<std::streamoff>(h.placement_offset));
    if (!in.read(reinterpret_cast<char*>(placement.data()), static_cast<std::streamsize>(placement.size() * sizeof(uint32_t)))) {
        set_reason(reason, "truncated placement table");
        return std::nullopt;
    }
    if (!is_permutation_of_range(placement, h.num_expert)) {
        set_reason(reason, "corrupt placement table");
        return std::nullopt;
    }
    layout._placement = std::move(placement);
    return layout;
}

PackBuildResult build_pack(const std::filesystem::path& dir,
                           const PackLayerSpec& spec,
                           const PackTensorSource& source,
                           const std::vector<uint32_t>& placement) {
    PackBuildResult result;
    result.path = dir / spec.file_name();
    if (auto existing = PackLayout::open(result.path, spec)) {
        if ((placement.empty() || existing->placement() == placement) && verify_pack(result.path, spec) == 0) {
            result.reused_existing = true;
            return result;
        }
    }

    auto layout = PackLayout::plan(spec);
    if (!placement.empty()) {
        layout.set_placement(placement);
    }

    std::filesystem::create_directories(dir);
    std::ostringstream tmp_name;
    tmp_name << result.path.filename().string() << ".tmp-" << std::hex << std::random_device{}() << std::random_device{}();
    const auto tmp_path = dir / tmp_name.str();

    std::vector<uint64_t> checksums(spec.num_expert, 0);
    try {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("OTD pack: cannot create " + tmp_path.string());
        }
        std::vector<char> zeros(layout.header().records_offset, 0);
        out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));

        std::vector<uint32_t> expert_at(spec.num_expert);
        for (uint32_t e = 0; e < spec.num_expert; e++) {
            expert_at[layout.placement()[e]] = e;
        }

        const uint64_t payload = layout.record_payload_bytes();
        std::vector<uint8_t> record(layout.record_stride());
        std::vector<uint8_t> tensor;
        for (uint64_t pos = 0; pos < spec.num_expert; pos++) {
            const uint32_t expert = expert_at[pos];
            std::fill(record.begin(), record.end(), uint8_t{0});
            for (size_t t = 0; t < pack_tensor_count; t++) {
                const uint64_t size = spec.tensor_bytes[t];
                if (size == 0) {
                    continue;
                }
                tensor.clear();
                source(expert, t, tensor);
                if (tensor.size() != size) {
                    throw std::runtime_error("OTD pack: tensor source returned an unexpected size");
                }
                std::memcpy(record.data() + layout.tensor_offset_in_record(t), tensor.data(), size);
            }
            checksums[expert] = fnv_update(fnv_offset, record.data(), payload);
            out.write(reinterpret_cast<const char*>(record.data()), static_cast<std::streamsize>(record.size()));
            result.bytes_written += record.size();
        }

        out.seekp(0);
        out.write(reinterpret_cast<const char*>(&layout.header()), sizeof(PackHeader));
        out.seekp(static_cast<std::streamoff>(layout.header().placement_offset));
        out.write(reinterpret_cast<const char*>(layout.placement().data()), static_cast<std::streamsize>(spec.num_expert * sizeof(uint32_t)));
        out.seekp(static_cast<std::streamoff>(layout.header().checksums_offset));
        out.write(reinterpret_cast<const char*>(checksums.data()), static_cast<std::streamsize>(checksums.size() * sizeof(uint64_t)));
        out.flush();
        if (!out) {
            throw std::runtime_error("OTD pack: write failed for " + tmp_path.string());
        }
        out.close();
        result.bytes_written += layout.header().records_offset;
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        throw;
    }

    std::error_code ec;
    std::filesystem::remove(result.path, ec);
    std::filesystem::rename(tmp_path, result.path, ec);
    if (ec) {
        // A concurrent builder may have published the same content first.
        std::filesystem::remove(tmp_path);
        if (!PackLayout::open(result.path, spec)) {
            throw std::runtime_error("OTD pack: cannot publish " + result.path.string() + ": " + ec.message());
        }
        result.reused_existing = true;
    }
    return result;
}

uint64_t verify_pack(const std::filesystem::path& path, const PackLayerSpec& spec, std::string* reason) {
    auto layout = PackLayout::open(path, spec, reason);
    if (!layout) {
        return spec.num_expert == 0 ? 1 : spec.num_expert;
    }
    std::ifstream in(path, std::ios::binary);
    std::vector<uint64_t> checksums(spec.num_expert);
    in.seekg(static_cast<std::streamoff>(layout->header().checksums_offset));
    in.read(reinterpret_cast<char*>(checksums.data()), static_cast<std::streamsize>(checksums.size() * sizeof(uint64_t)));

    const uint64_t payload = layout->record_payload_bytes();
    std::vector<uint8_t> record(payload);
    uint64_t corrupt = 0;
    for (uint64_t e = 0; e < spec.num_expert; e++) {
        in.seekg(static_cast<std::streamoff>(layout->record_offset(e)));
        if (!in.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(payload)) ||
            fnv_update(fnv_offset, record.data(), payload) != checksums[e]) {
            corrupt++;
            in.clear();
        }
    }
    if (corrupt) {
        set_reason(reason, std::to_string(corrupt) + " corrupt records");
    }
    return corrupt;
}

}  // namespace ov::intel_gpu::ocl::moe_otd
