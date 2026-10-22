// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>
#include <vector>

#include "ocl_v2/moe/moe_otd_trace.hpp"
#include "test_utils/test_utils.h"

using ov::intel_gpu::ocl::moe_otd::AcquireKind;
using ov::intel_gpu::ocl::moe_otd::ExpertLoadStats;
using ov::intel_gpu::ocl::moe_otd::fnv1a64;
using ov::intel_gpu::ocl::moe_otd::LatencyHistogram;
using ov::intel_gpu::ocl::moe_otd::TraceWriter;

namespace {

struct CapturedTrace {
    std::ostringstream* stream = nullptr;
    std::unique_ptr<TraceWriter> writer;
};

CapturedTrace make_trace() {
    auto out = std::make_unique<std::ostringstream>();
    CapturedTrace captured;
    captured.stream = out.get();
    captured.writer = std::make_unique<TraceWriter>(std::move(out));
    return captured;
}

std::vector<std::string> lines_of(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        lines.push_back(line);
    }
    return lines;
}

// Drops the t_ns column (third field) of C/E records so expectations are deterministic.
std::string without_time(const std::string& line) {
    if (line.empty() || (line[0] != 'C' && line[0] != 'E')) {
        return line;
    }
    const auto first = line.find(',');
    const auto second = line.find(',', first + 1);
    const auto third = line.find(',', second + 1);
    return line.substr(0, second) + line.substr(third);
}

}  // namespace

TEST(moe_otd_trace, fnv1a64_reference_values) {
    ASSERT_EQ(fnv1a64(""), 14695981039346656037ULL);
    ASSERT_EQ(fnv1a64("a"), 0xaf63dc4c8601ec8cULL);
    ASSERT_NE(fnv1a64("layer.0"), fnv1a64("layer.1"));
}

TEST(moe_otd_trace, histogram_empty_reports_zero) {
    LatencyHistogram h;
    ASSERT_EQ(h.count(), 0U);
    ASSERT_EQ(h.quantile_upper_bound_us(0.5), 0U);
}

TEST(moe_otd_trace, histogram_power_of_two_buckets) {
    LatencyHistogram h;
    h.record_ns(500);      // 0 us      -> bucket 0, upper bound 1
    h.record_ns(1'000);    // 1 us      -> bucket 1, upper bound 2
    h.record_ns(3'000);    // 3 us      -> bucket 2, upper bound 4
    h.record_ns(100'000);  // 100 us    -> bucket 7, upper bound 128
    ASSERT_EQ(h.count(), 4U);
    const auto buckets = h.buckets();
    ASSERT_EQ(buckets[0], 1U);
    ASSERT_EQ(buckets[1], 1U);
    ASSERT_EQ(buckets[2], 1U);
    ASSERT_EQ(buckets[7], 1U);
    ASSERT_EQ(h.quantile_upper_bound_us(0.25), 1U);
    ASSERT_EQ(h.quantile_upper_bound_us(0.5), 2U);
    ASSERT_EQ(h.quantile_upper_bound_us(0.75), 4U);
    ASSERT_EQ(h.quantile_upper_bound_us(1.0), 128U);
}

TEST(moe_otd_trace, histogram_saturates_last_bucket) {
    LatencyHistogram h;
    h.record_ns(~uint64_t{0});
    ASSERT_EQ(h.buckets()[LatencyHistogram::bucket_count - 1], 1U);
}

TEST(moe_otd_trace, header_and_layer_records) {
    auto trace = make_trace();
    ASSERT_EQ(trace.writer->register_layer("moe.0", "model.bin", 16, 128, 8), 0U);
    ASSERT_EQ(trace.writer->register_layer("moe.1", "model.bin", 16, 128, 8), 1U);
    trace.writer->flush();

    const auto lines = lines_of(trace.stream->str());
    ASSERT_EQ(lines.size(), 3U);
    ASSERT_EQ(lines[0], "# moe_otd_trace v1");
    std::ostringstream expected;
    expected << "L,0," << std::hex << fnv1a64("moe.0") << ',' << fnv1a64("model.bin") << std::dec << ",16,128,8";
    ASSERT_EQ(lines[1], expected.str());
    // Primitive names and weight paths are only ever written as hashes.
    ASSERT_EQ(trace.stream->str().find("model.bin"), std::string::npos);
    ASSERT_EQ(trace.stream->str().find("moe.0"), std::string::npos);
}

TEST(moe_otd_trace, call_record_precedes_its_expert_records) {
    auto trace = make_trace();
    const auto layer = trace.writer->register_layer("moe.0", "model.bin", 4, 64, 2);
    const auto call = trace.writer->begin_call(layer, AcquireKind::simultaneous, 4, {7, 3});

    ExpertLoadStats load;
    load.bytes = 4096;
    load.reads = 9;
    load.read_ns = 100;
    load.transform_ns = 20;
    load.copy_ns = 30;
    trace.writer->expert(call, layer, 7, 0, false, std::nullopt, load);
    trace.writer->expert(call, layer, 3, 1, true, std::nullopt, {});
    trace.writer->end_call(call, false);
    trace.writer->flush();

    const auto lines = lines_of(trace.stream->str());
    ASSERT_EQ(lines.size(), 5U);
    ASSERT_EQ(without_time(lines[2]), "C,0,0,S,4,ok,7 3");
    ASSERT_EQ(without_time(lines[3]), "E,0,0,7,0,0,-1,4096,9,100,20,30");
    ASSERT_EQ(without_time(lines[4]), "E,0,0,3,1,1,-1,0,0,0,0,0");
}

TEST(moe_otd_trace, overflow_and_eviction_are_recorded) {
    auto trace = make_trace();
    const auto layer = trace.writer->register_layer("moe.0", "model.bin", 1, 64, 2);
    const auto call = trace.writer->begin_call(layer, AcquireKind::simultaneous, 2, {5, 6});
    trace.writer->expert(call, layer, 5, 0, false, size_t{4}, {});
    trace.writer->end_call(call, true);
    trace.writer->flush();

    const auto lines = lines_of(trace.stream->str());
    ASSERT_EQ(lines.size(), 4U);
    ASSERT_EQ(without_time(lines[2]), "C,0,0,S,2,overflow,5 6");
    ASSERT_EQ(without_time(lines[3]), "E,0,0,5,0,0,4,0,0,0,0,0");
}

TEST(moe_otd_trace, interleaved_calls_are_emitted_whole_in_completion_order) {
    auto trace = make_trace();
    const auto l0 = trace.writer->register_layer("moe.0", "target.bin", 2, 8, 1);
    const auto l1 = trace.writer->register_layer("moe.0", "draft.bin", 2, 8, 1);
    const auto a = trace.writer->begin_call(l0, AcquireKind::one, 1, {1});
    const auto b = trace.writer->begin_call(l1, AcquireKind::one, 1, {2});
    trace.writer->expert(b, l1, 2, 0, false, std::nullopt, {});
    trace.writer->expert(a, l0, 1, 0, false, std::nullopt, {});
    trace.writer->end_call(b, false);
    trace.writer->end_call(a, false);
    trace.writer->flush();

    const auto lines = lines_of(trace.stream->str());
    ASSERT_EQ(lines.size(), 7U);
    ASSERT_EQ(without_time(lines[3]), "C,1,1,O,1,ok,2");
    ASSERT_EQ(without_time(lines[4]), "E,1,1,2,0,0,-1,0,0,0,0,0");
    ASSERT_EQ(without_time(lines[5]), "C,0,0,O,1,ok,1");
    ASSERT_EQ(without_time(lines[6]), "E,0,0,1,0,0,-1,0,0,0,0,0");
}

TEST(moe_otd_trace, unknown_call_is_ignored) {
    auto trace = make_trace();
    trace.writer->expert(42, 0, 1, 0, false, std::nullopt, {});
    trace.writer->end_call(42, false);
    trace.writer->flush();
    ASSERT_EQ(lines_of(trace.stream->str()).size(), 1U);
}
