// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <thread>

namespace {
using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
namespace otd = ov::intel_gpu::ocl::moe_otd;

struct NativePackFixture {
    std::filesystem::path directory;
    std::filesystem::path weights;
    MOECompressed::Config config{};
    std::vector<size_t> offsets;
    otd::RoutedTensorLayouts layouts;

    NativePackFixture() {
        std::random_device random;
        directory = std::filesystem::temp_directory_path() /
                    ("otd_native_gpu_" + std::to_string(random()) + std::to_string(random()));
        std::filesystem::create_directories(directory);
        weights = directory / "source.bin";
        config.num_expert = 4;
        config.top_k = 2;
        config.hidden_size = 8;
        config.inter_size = 8;
        config.group_size = 4;
        config.has_zp = true;
        std::ofstream file(weights, std::ios::binary);
        size_t offset = 0;
        for (size_t tensor = 0; tensor < 9; ++tensor) {
            const auto type = tensor >= 3 && tensor < 6 ? data_types::f16 : data_types::u4;
            const int64_t groups_or_hidden = tensor < 3 ? 8 : 2;
            layouts.emplace_back(type, format::bfyx, cldnn::tensor{4, 1, 8, groups_or_hidden});
            offsets.push_back(offset);
            const size_t bytes = layouts.back().bytes_count();
            const size_t per_expert = bytes / 4;
            std::vector<uint8_t> payload(bytes);
            for (size_t i = 0; i < bytes; ++i)
                payload[i] = static_cast<uint8_t>(tensor * 17 + (i / per_expert) * 31 + i % per_expert);
            file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
            offset += bytes;
        }
        if (!file) throw std::runtime_error("native OTD fixture write failed");
    }
    ~NativePackFixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    std::array<memory::ptr, 9> allocate() const {
        std::array<memory::ptr, 9> buffers;
        for (size_t tensor = 0; tensor < buffers.size(); ++tensor) {
            buffers[tensor] = get_test_engine().allocate_memory(layouts[tensor], allocation_type::usm_device);
            std::vector<uint8_t> sentinel(layouts[tensor].bytes_count(), 0xa5);
            buffers[tensor]->copy_from(get_test_stream(), sentinel.data(), 0, 0, sentinel.size(), true);
        }
        return buffers;
    }
    static moe_weights arena(const std::array<memory::ptr, 9>& b) {
        moe_weights result;
        result.gate_w = b[0]; result.up_w = b[1]; result.down_w = b[2];
        result.gate_s = b[3]; result.up_s = b[4]; result.down_s = b[5];
        result.gate_z = b[6]; result.up_z = b[7]; result.down_z = b[8];
        return result;
    }
    static std::vector<uint8_t> read(const memory::ptr& memory) {
        std::vector<uint8_t> result(memory->get_layout().bytes_count());
        memory->copy_to(get_test_stream(), result.data(), 0, 0, result.size(), true);
        return result;
    }
};
}  // namespace

TEST(moe_otd_native_gpu, nonidentity_pack_matches_plain_demand_copies_through_eviction) {
    NativePackFixture fixture;
    auto plain_buffers = fixture.allocate();
    auto packed_buffers = fixture.allocate();
    auto plain_arena = fixture.arena(plain_buffers);
    auto packed_arena = fixture.arena(packed_buffers);
    otd::ParallelWeightReader reader(fixture.weights);
    const auto spec = otd::make_pack_spec(fixture.config, fixture.offsets, fixture.layouts, reader);
    const auto built = otd::build_layer_pack(fixture.directory, spec, fixture.config, fixture.layouts, reader, {3, 1, 0, 2});
    ASSERT_GT(built.bytes_written, 0u);
    OffloadExpertWeightProvider plain(2, fixture.config, fixture.offsets, fixture.weights);
    OffloadExpertWeightProvider packed(2, fixture.config, fixture.offsets, fixture.weights);
    plain.bind(plain_arena);
    packed.bind(packed_arena);
    packed.configure_pack(otd::PackMode::require, fixture.directory, fixture.layouts);
    ASSERT_TRUE(packed.uses_pack());
    auto& stream = get_test_stream();
    for (uint32_t expert : {3u, 1u, 2u, 3u, 0u, 0u}) {
        auto plain_scope = plain.enter_execution([&] { stream.finish(); });
        auto packed_scope = packed.enter_execution([&] { stream.finish(); });
        ASSERT_EQ(plain.acquire_one(expert, stream), packed.acquire_one(expert, stream));
        for (size_t tensor = 0; tensor < 9; ++tensor)
            EXPECT_EQ(fixture.read(plain_buffers[tensor]), fixture.read(packed_buffers[tensor])) << tensor << ":" << expert;
    }
    ASSERT_GT(packed.cache().evictions(), 0u);
    const auto evictions = packed.cache().evictions();
    const auto before = fixture.read(packed_buffers[0]);
    EXPECT_FALSE(packed.try_acquire_simultaneous({0, 1, 2}, stream));
    EXPECT_EQ(evictions, packed.cache().evictions());
    EXPECT_EQ(before, fixture.read(packed_buffers[0]));
}

TEST(moe_otd_native_gpu, corrupt_record_never_changes_device_slot_or_publishes_hit) {
    NativePackFixture fixture;
    otd::ParallelWeightReader reader(fixture.weights);
    const auto spec = otd::make_pack_spec(fixture.config, fixture.offsets, fixture.layouts, reader);
    const auto built = otd::build_layer_pack(fixture.directory, spec, fixture.config, fixture.layouts, reader);
    const auto layout = otd::PackLayout::open(built.path, spec);
    ASSERT_TRUE(layout);
    {
        std::fstream file(built.path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file);
        file.seekg(static_cast<std::streamoff>(layout->record_offset(3)));
        char byte = 0;
        file.read(&byte, 1);
        byte ^= 1;
        file.seekp(static_cast<std::streamoff>(layout->record_offset(3)));
        file.write(&byte, 1);
        ASSERT_TRUE(file);
    }
    auto buffers = fixture.allocate();
    auto arena = fixture.arena(buffers);
    OffloadExpertWeightProvider provider(2, fixture.config, fixture.offsets, fixture.weights);
    provider.bind(arena);
    provider.configure_pack(otd::PackMode::require, fixture.directory, fixture.layouts);
    auto& stream = get_test_stream();
    auto scope = provider.enter_execution([&] { stream.finish(); });
    EXPECT_THROW(provider.acquire_one(3, stream), ov::Exception);
    EXPECT_FALSE(provider.cache().lookup(3).hit);
    for (const auto& buffer : buffers)
        EXPECT_EQ(fixture.read(buffer), std::vector<uint8_t>(buffer->get_layout().bytes_count(), 0xa5));
}

static void check_shared_provider_terminal_read(bool cancel_reader) {
    NativePackFixture fixture;
    auto buffers = fixture.allocate();
    auto arena = fixture.arena(buffers);
    auto config = get_test_default_config(get_test_engine());
    config.set_property(ov::intel_gpu::queue_type(ov::intel_gpu::QueueTypes::in_order));
    auto stream_owner = get_test_engine().create_stream(config);
    auto& stream = *stream_owner;
    auto& queue = dynamic_cast<ocl::ocl_stream&>(stream).get_cl_queue();
    auto& context = dynamic_cast<ocl::ocl_engine&>(get_test_engine()).get_cl_context();
    OffloadExpertWeightProvider provider(1, fixture.config, fixture.offsets, fixture.weights);
    provider.bind(arena);
    ASSERT_EQ(provider.acquire_one(0, stream), 0u);
    const size_t bytes = fixture.layouts[0].bytes_count() / fixture.config.num_expert;
    std::vector<uint8_t> expected(bytes), observed(bytes);
    buffers[0]->copy_to(stream, expected.data(), 0, 0, bytes, true);
    cl::UserEvent producer_gate(context);
    cl::Event terminal_consumer;
    std::promise<void> queued, attempted, entered;
    auto queued_future = queued.get_future();
    auto attempted_future = attempted.get_future();
    auto entered_future = entered.get_future();
    std::exception_ptr reader_error, writer_error;
    std::thread reader([&] {
        bool published = false;
        try {
            auto scope = provider.enter_execution([&] { stream.finish(); });
            const cl_event gate = producer_gate.get();
            if (clEnqueueBarrierWithWaitList(queue.get(), 1, &gate, nullptr) != CL_SUCCESS)
                throw std::runtime_error("native GPU test could not gate the consumer");
            buffers[0]->copy_to(stream, observed.data(), 0, 0, bytes, false);
            cl_event terminal = nullptr;
            if (clEnqueueMarkerWithWaitList(queue.get(), 0, nullptr, &terminal) != CL_SUCCESS)
                throw std::runtime_error("native GPU test could not retain the terminal event");
            terminal_consumer = cl::Event(terminal, false);
            stream.flush();
            queued.set_value();
            published = true;
            if (cancel_reader) throw std::runtime_error("cancelled GPU consumer");
            // Scope exit drains while the real GPU read is blocked on the
            // explicit OpenCL user event. It must retain the provider lock.
        } catch (...) {
            reader_error = std::current_exception();
            if (!published) queued.set_exception(reader_error);
        }
    });
    const bool queued_in_time = queued_future.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    bool queued_ok = false;
    if (queued_in_time) {
        try { queued_future.get(); queued_ok = true; } catch (...) {}
    }
    std::thread writer;
    bool remained_blocked = false, terminal_was_pending = false;
    size_t evictions_while_reading = 0;
    if (queued_ok) {
        terminal_was_pending = terminal_consumer.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() != CL_COMPLETE;
        writer = std::thread([&] {
            bool published = false;
            try {
                attempted.set_value();
                auto scope = provider.enter_execution([&] { stream.finish(); });
                entered.set_value();
                published = true;
                provider.acquire_one(1, stream);
            } catch (...) {
                writer_error = std::current_exception();
                if (!published) entered.set_exception(writer_error);
            }
        });
        attempted_future.wait();
        remained_blocked = entered_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
        evictions_while_reading = provider.cache().evictions();
    }
    // Always release and join before any fatal assertion can destroy buffers.
    const auto release = clSetUserEventStatus(producer_gate.get(), CL_COMPLETE);
    reader.join();
    if (writer.joinable()) writer.join();
    ASSERT_EQ(release, CL_SUCCESS);
    ASSERT_TRUE(queued_in_time && queued_ok);
    if (cancel_reader) {
        ASSERT_TRUE(reader_error);
        try { std::rethrow_exception(reader_error); }
        catch (const std::runtime_error& error) { EXPECT_STREQ(error.what(), "cancelled GPU consumer"); }
        catch (...) { FAIL() << "unexpected cancellation exception"; }
    } else {
        ASSERT_FALSE(reader_error);
    }
    ASSERT_FALSE(writer_error);
    EXPECT_TRUE(terminal_was_pending);
    EXPECT_TRUE(remained_blocked);
    EXPECT_EQ(evictions_while_reading, 0u);
    EXPECT_EQ(terminal_consumer.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(), CL_COMPLETE);
    EXPECT_EQ(observed, expected);
    EXPECT_EQ(provider.cache().evictions(), 1u);
}

TEST(moe_otd_native_gpu, shared_provider_waits_for_terminal_device_read_before_replacement) {
    check_shared_provider_terminal_read(false);
}

TEST(moe_otd_native_gpu, cancellation_drains_terminal_device_read_before_replacement) {
    check_shared_provider_terminal_read(true);
}
