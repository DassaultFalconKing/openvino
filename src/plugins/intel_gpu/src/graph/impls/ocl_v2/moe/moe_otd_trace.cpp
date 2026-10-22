// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_trace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <sstream>

namespace ov::intel_gpu::ocl::moe_otd {

void LatencyHistogram::record_ns(uint64_t ns) {
    uint64_t us = ns / 1000;
    size_t bucket = 0;
    while (us != 0 && bucket + 1 < bucket_count) {
        us >>= 1;
        bucket++;
    }
    _buckets[bucket].fetch_add(1, std::memory_order_relaxed);
}

uint64_t LatencyHistogram::count() const {
    uint64_t total = 0;
    for (const auto& b : _buckets) {
        total += b.load(std::memory_order_relaxed);
    }
    return total;
}

std::array<uint64_t, LatencyHistogram::bucket_count> LatencyHistogram::buckets() const {
    std::array<uint64_t, bucket_count> result{};
    for (size_t i = 0; i < bucket_count; i++) {
        result[i] = _buckets[i].load(std::memory_order_relaxed);
    }
    return result;
}

uint64_t LatencyHistogram::quantile_upper_bound_us(double q) const {
    const auto snapshot = buckets();
    uint64_t total = 0;
    for (auto v : snapshot) {
        total += v;
    }
    if (total == 0) {
        return 0;
    }
    q = std::clamp(q, 0.0, 1.0);
    const auto rank = std::max<uint64_t>(1, static_cast<uint64_t>(q * static_cast<double>(total) + 0.999999));
    uint64_t seen = 0;
    for (size_t i = 0; i < bucket_count; i++) {
        seen += snapshot[i];
        if (seen >= rank) {
            return uint64_t{1} << i;
        }
    }
    return uint64_t{1} << (bucket_count - 1);
}

uint64_t fnv1a64(std::string_view text) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

TraceWriter::TraceWriter(std::unique_ptr<std::ostream> out) : _out(std::move(out)) {
    _origin_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (_out) {
        *_out << "# moe_otd_trace v1\n";
    }
}

TraceWriter::~TraceWriter() {
    close();
}

uint64_t TraceWriter::now_ns() const {
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    return static_cast<uint64_t>(now - _origin_ns);
}

uint32_t TraceWriter::register_layer(std::string_view layer_name, std::string_view model_identity, size_t capacity, size_t num_expert, size_t top_k) {
    std::lock_guard<std::mutex> lock(_mutex);
    const uint32_t layer = _next_layer++;
    if (_out && !_closed) {
        *_out << "L," << layer << ',' << std::hex << fnv1a64(layer_name) << ',' << fnv1a64(model_identity) << std::dec << ',' << capacity << ',' << num_expert
              << ',' << top_k << '\n';
    }
    return layer;
}

uint64_t TraceWriter::begin_call(uint32_t layer, AcquireKind kind, size_t routed, const std::vector<uint32_t>& unique_experts) {
    const auto t = now_ns();
    std::lock_guard<std::mutex> lock(_mutex);
    if (_closed) {
        return _next_call++;
    }
    PendingCall call;
    call.id = _next_call++;
    call.t_ns = t;
    call.layer = layer;
    call.kind = kind;
    call.routed = routed;
    call.unique_experts = unique_experts;
    _pending.push_back(std::move(call));
    return _pending.back().id;
}

void TraceWriter::expert(uint64_t call, uint32_t layer, uint32_t expert, size_t slot, bool hit, std::optional<size_t> evicted, const ExpertLoadStats& load) {
    const auto t = now_ns();
    std::ostringstream line;
    line << "E," << call << ',' << t << ',' << layer << ',' << expert << ',' << slot << ',' << (hit ? 1 : 0) << ',';
    if (evicted) {
        line << *evicted;
    } else {
        line << -1;
    }
    line << ',' << load.bytes << ',' << load.reads << ',' << load.read_ns << ',' << load.transform_ns << ',' << load.copy_ns << '\n';

    std::lock_guard<std::mutex> lock(_mutex);
    auto it = std::find_if(_pending.begin(), _pending.end(), [&](const PendingCall& c) {
        return c.id == call;
    });
    if (it != _pending.end()) {
        it->expert_lines.push_back(line.str());
    }
}

void TraceWriter::end_call(uint64_t call, bool overflow) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = std::find_if(_pending.begin(), _pending.end(), [&](const PendingCall& c) {
        return c.id == call;
    });
    if (it == _pending.end()) {
        return;
    }
    if (_out) {
        write_call_unlocked(*it, overflow ? "overflow" : "ok", {});
        _calls_written++;
        _experts_written += it->expert_lines.size();
    }
    _pending.erase(it);
}

void TraceWriter::write_call_unlocked(const PendingCall& call, std::string_view status, std::string_view prefix) {
    auto& out = *_out;
    out << prefix << "C," << call.id << ',' << call.t_ns << ',' << call.layer << ',' << static_cast<char>(call.kind) << ',' << call.routed << ','
        << status << ',';
    for (size_t i = 0; i < call.unique_experts.size(); i++) {
        if (i != 0) {
            out << ' ';
        }
        out << call.unique_experts[i];
    }
    out << '\n';
    for (const auto& line : call.expert_lines) {
        out << prefix << line;
    }
}

void TraceWriter::flush() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_out) {
        _out->flush();
    }
}

void TraceWriter::close() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_closed) {
        return;
    }
    _closed = true;
    if (!_out) {
        _pending.clear();
        return;
    }
    uint64_t incomplete_experts = 0;
    for (const auto& call : _pending) {
        write_call_unlocked(call, "incomplete", "# incomplete ");
        incomplete_experts += call.expert_lines.size();
    }
    *_out << "# end,calls=" << _calls_written << ",experts=" << _experts_written << ",incomplete_calls=" << _pending.size()
          << ",incomplete_experts=" << incomplete_experts << '\n';
    _out->flush();
    _pending.clear();
}

TraceWriter* get_trace_writer() {
    static std::unique_ptr<TraceWriter> writer = []() -> std::unique_ptr<TraceWriter> {
        const char* path = std::getenv("MOE_OTD_TRACE");
        if (path == nullptr || *path == '\0') {
            return nullptr;
        }
        auto file = std::make_unique<std::ofstream>(path, std::ios::out | std::ios::trunc);
        if (!file->is_open()) {
            return nullptr;
        }
        return std::make_unique<TraceWriter>(std::move(file));
    }();
    return writer.get();
}

}  // namespace ov::intel_gpu::ocl::moe_otd
