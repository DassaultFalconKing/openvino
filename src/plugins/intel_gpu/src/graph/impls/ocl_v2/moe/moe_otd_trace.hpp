// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Dependency-free OTD observability primitives: per-load statistics, a lock-free
// latency histogram and an expert-routing trace writer. Nothing here changes
// cache or routing behaviour; everything is inert unless explicitly enabled.
namespace ov::intel_gpu::ocl::moe_otd {

// Accumulated cost of streaming expert tensors from disk into device slots.
struct ExpertLoadStats {
    uint64_t bytes = 0;
    uint64_t reads = 0;
    uint64_t read_ns = 0;
    uint64_t transform_ns = 0;
    uint64_t copy_ns = 0;

    ExpertLoadStats& operator+=(const ExpertLoadStats& rhs) {
        bytes += rhs.bytes;
        reads += rhs.reads;
        read_ns += rhs.read_ns;
        transform_ns += rhs.transform_ns;
        copy_ns += rhs.copy_ns;
        return *this;
    }
};

// Power-of-two microsecond buckets: bucket 0 holds [0, 1) us, bucket i holds [2^(i-1), 2^i) us.
class LatencyHistogram {
public:
    static constexpr size_t bucket_count = 32;

    void record_ns(uint64_t ns);
    uint64_t count() const;
    // Upper bound (in us) of the bucket containing the requested quantile; 0 when empty.
    uint64_t quantile_upper_bound_us(double q) const;
    std::array<uint64_t, bucket_count> buckets() const;

private:
    std::array<std::atomic<uint64_t>, bucket_count> _buckets{};
};

// Stable, non-cryptographic 64-bit FNV-1a. Used to tag layers/models in traces
// without writing primitive names or file-system paths.
uint64_t fnv1a64(std::string_view text);

enum class AcquireKind : char { simultaneous = 'S', one = 'O' };

// Line-oriented trace, format "moe_otd_trace v1":
//   L,<layer>,<layer_key>,<model_key>,<capacity>,<num_expert>,<top_k>
//   C,<call>,<t_ns>,<layer>,<kind>,<routed>,<status>,<unique experts, space separated, first-occurrence order>
//   E,<call>,<t_ns>,<layer>,<expert>,<slot>,<hit>,<evicted or -1>,<bytes>,<reads>,<read_ns>,<transform_ns>,<copy_ns>
// E records describe every expert the cache processed for call <call>. An overflowing
// simultaneous call is rejected before the cache is touched and therefore has no E records.
// Timestamps are steady-clock nanoseconds since the writer was opened. No prompt data is recorded.
// <bytes> and <reads> are logical: what the plugin asked to read, not what the device transferred.
//
// close() appends completeness records as '#' comment lines, which v1 readers skip:
//   # incomplete C,<call>,<t_ns>,<layer>,<kind>,<routed>,incomplete,<unique experts>
//   # incomplete E,...                       (same fields as E)
//   # end,calls=<C records>,experts=<E records>,incomplete_calls=<n>,incomplete_experts=<n>
// An incomplete call was begun but never ended, e.g. because loading an expert threw; its expert
// records still changed the cache. A trace without the end record was not closed (killed process,
// or the writer is still open) and may be missing its tail.
class TraceWriter {
public:
    explicit TraceWriter(std::unique_ptr<std::ostream> out);
    // Calls close().
    ~TraceWriter();

    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;

    uint32_t register_layer(std::string_view layer_name, std::string_view model_identity, size_t capacity, size_t num_expert, size_t top_k);

    uint64_t begin_call(uint32_t layer, AcquireKind kind, size_t routed, const std::vector<uint32_t>& unique_experts);
    void end_call(uint64_t call, bool overflow);

    void expert(uint64_t call, uint32_t layer, uint32_t expert, size_t slot, bool hit, std::optional<size_t> evicted, const ExpertLoadStats& load);

    void flush();

    // Writes pending calls as incomplete, then the end record, and flushes. Idempotent; records
    // arriving afterwards are dropped so nothing follows the end record.
    void close();

private:
    uint64_t now_ns() const;

    struct PendingCall {
        uint64_t id = 0;
        uint64_t t_ns = 0;
        uint32_t layer = 0;
        AcquireKind kind = AcquireKind::one;
        size_t routed = 0;
        std::vector<uint32_t> unique_experts;
        std::vector<std::string> expert_lines;
    };

    // Requires _mutex and _out. Writes the C record and its E records, each prefixed with `prefix`.
    void write_call_unlocked(const PendingCall& call, std::string_view status, std::string_view prefix);

    std::mutex _mutex;
    std::unique_ptr<std::ostream> _out;
    uint32_t _next_layer = 0;
    uint64_t _next_call = 0;
    std::vector<PendingCall> _pending;
    int64_t _origin_ns = 0;
    bool _closed = false;
    uint64_t _calls_written = 0;
    uint64_t _experts_written = 0;
};

// Returns the process-wide trace writer when MOE_OTD_TRACE=<file> is set, otherwise nullptr.
TraceWriter* get_trace_writer();

}  // namespace ov::intel_gpu::ocl::moe_otd
