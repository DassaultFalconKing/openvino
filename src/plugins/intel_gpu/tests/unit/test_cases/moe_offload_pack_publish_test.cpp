// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <atomic>
#include <fstream>
#include <iterator>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ocl_v2/moe/moe_otd_pack.hpp"
#include "test_utils/test_utils.h"

using namespace ov::intel_gpu::ocl::moe_otd;

// Publication of a built pack (build_pack): the destination must never disappear or be seen
// partially written, an existing destination is only replaced by a complete verified file, and a
// failed publication leaves the previous file as it was. Windows and POSIX differ in whether a
// replacement succeeds while a reader holds the destination open, so these tests assert the
// invariant that must hold on both rather than one platform's outcome.

namespace {

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        path = std::filesystem::temp_directory_path() / ("moe_otd_publish_test_" + std::to_string(rd()) + std::to_string(rd()));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    std::filesystem::path path;
};

PackLayerSpec make_spec() {
    PackLayerSpec spec;
    spec.num_expert = 6;
    spec.source_file_size = 1 << 20;
    spec.source_sample_hash = 0x5678;
    spec.tensor_bytes = {1000, 1000, 777, 40, 40, 31, 20, 20, 0};
    for (size_t t = 0; t < pack_tensor_count; t++) {
        spec.source_offsets[t] = 4096 * (t + 1);
    }
    spec.layout_params = {2048, 512, 128, 1};
    return spec;
}

PackTensorSource pattern_source(const PackLayerSpec& spec) {
    return [spec](uint32_t expert, size_t tensor, std::vector<uint8_t>& out) {
        out.resize(spec.tensor_bytes[tensor]);
        for (size_t i = 0; i < out.size(); i++) {
            out[i] = static_cast<uint8_t>((expert * 131 + tensor * 17 + i * 7) & 0xFF);
        }
    };
}

const std::vector<uint32_t> reversed_placement = {5, 4, 3, 2, 1, 0};

std::vector<char> file_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
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

std::vector<std::string> entries(const std::filesystem::path& dir) {
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        names.push_back(entry.path().filename().string());
    }
    return names;
}

}  // namespace

TEST(moe_otd_pack_publish, replaces_unopened_corrupt_destination) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    auto layout = PackLayout::open(built.path, spec);
    ASSERT_TRUE(layout.has_value());
    flip_byte(built.path, layout->record_offset(2) + 10);
    ASSERT_EQ(verify_pack(built.path, spec), 1U);

    const auto rebuilt = build_pack(dir.path, spec, pattern_source(spec));
    ASSERT_FALSE(rebuilt.reused_existing);
    ASSERT_EQ(verify_pack(built.path, spec), 0U);
    ASSERT_EQ(entries(dir.path), std::vector<std::string>{spec.file_name()});
}

TEST(moe_otd_pack_publish, directory_at_destination_is_left_alone_and_fails) {
    // An unexpected non-file at the destination must not be deleted to make room.
    TempDir dir;
    const auto spec = make_spec();
    const auto dest = dir.path / spec.file_name();
    std::filesystem::create_directory(dest);

    ASSERT_THROW(build_pack(dir.path, spec, pattern_source(spec)), std::runtime_error);
    ASSERT_TRUE(std::filesystem::is_directory(dest));
    ASSERT_EQ(entries(dir.path), std::vector<std::string>{spec.file_name()});
}

TEST(moe_otd_pack_publish, corrupt_destination_held_open_is_never_reported_as_reused) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    auto layout = PackLayout::open(built.path, spec);
    ASSERT_TRUE(layout.has_value());
    flip_byte(built.path, layout->record_offset(4) + 3);
    const auto corrupt_bytes = file_bytes(built.path);

    // A serving process has the old pack open. On Windows this blocks replacing it; on POSIX the
    // rename replaces the directory entry while the reader keeps the old inode.
    std::ifstream reader(built.path, std::ios::binary);
    ASSERT_TRUE(reader.is_open());
    bool returned = false;
    PackBuildResult result;
    try {
        result = build_pack(dir.path, spec, pattern_source(spec));
        returned = true;
    } catch (const std::runtime_error&) {
    }
    reader.close();

    if (returned) {
        ASSERT_FALSE(result.reused_existing);
        ASSERT_EQ(verify_pack(built.path, spec), 0U);
    } else {
        ASSERT_EQ(file_bytes(built.path), corrupt_bytes);
    }
    ASSERT_EQ(entries(dir.path), std::vector<std::string>{spec.file_name()});
}

TEST(moe_otd_pack_publish, failed_replacement_keeps_previous_pack_and_never_returns_wrong_placement) {
    TempDir dir;
    const auto spec = make_spec();
    const auto built = build_pack(dir.path, spec, pattern_source(spec));
    const auto previous_bytes = file_bytes(built.path);

    std::ifstream reader(built.path, std::ios::binary);
    ASSERT_TRUE(reader.is_open());
    bool returned = false;
    try {
        build_pack(dir.path, spec, pattern_source(spec), reversed_placement);
        returned = true;
    } catch (const std::runtime_error&) {
    }
    reader.close();

    auto layout = PackLayout::open(built.path, spec);
    ASSERT_TRUE(layout.has_value());
    ASSERT_EQ(verify_pack(built.path, spec), 0U);
    if (returned) {
        ASSERT_EQ(layout->placement(), reversed_placement);
    } else {
        ASSERT_EQ(file_bytes(built.path), previous_bytes);
    }
    ASSERT_EQ(entries(dir.path), std::vector<std::string>{spec.file_name()});
}

TEST(moe_otd_pack_publish, concurrent_rebuilds_never_hide_the_destination_from_readers) {
    TempDir dir;
    const auto spec = make_spec();
    const auto dest = build_pack(dir.path, spec, pattern_source(spec)).path;

    std::atomic<bool> stop{false};
    std::atomic<size_t> reader_opens{0};
    size_t transient_open_failures = 0;
    std::vector<std::string> reader_failures;
    std::thread reader([&] {
        while (!stop.load()) {
            std::string reason;
            if (PackLayout::open(dest, spec, &reason)) {
                reader_opens++;
                continue;
            }
#ifdef _WIN32
            // MoveFileExW replacement briefly denies concurrent opens of the destination. That is
            // a loud, retryable failure; a missing, truncated or mixed file is not acceptable.
            if (reason == "cannot open") {
                transient_open_failures++;
                continue;
            }
#endif
            if (reader_failures.size() < 16) {
                reader_failures.push_back(reason);
            }
        }
    });

    // Alternating placements force every build to publish over an existing pack.
    constexpr size_t builders = 4;
    constexpr size_t rounds = 12;
    std::atomic<size_t> published{0};
    std::atomic<size_t> refused{0};
    std::vector<std::string> unexpected;
    std::mutex unexpected_mutex;
    std::vector<std::thread> threads;
    for (size_t b = 0; b < builders; b++) {
        threads.emplace_back([&, b] {
            for (size_t r = 0; r < rounds; r++) {
                const bool reversed = (b + r) % 2 == 1;
                try {
                    build_pack(dir.path, spec, pattern_source(spec), reversed ? reversed_placement : std::vector<uint32_t>{});
                    published++;
                } catch (const std::runtime_error& e) {
                    const std::string what = e.what();
                    if (what.find("cannot publish") != std::string::npos) {
                        refused++;
                    } else {
                        std::lock_guard<std::mutex> lock(unexpected_mutex);
                        unexpected.push_back(what);
                    }
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    stop = true;
    reader.join();

    ASSERT_TRUE(unexpected.empty()) << unexpected.front();
    ASSERT_TRUE(reader_failures.empty()) << reader_failures.size() << " failed opens, first: " << reader_failures.front();
    ASSERT_GT(reader_opens.load(), 0U);
    RecordProperty("transient_open_failures", static_cast<int>(transient_open_failures));
    ASSERT_EQ(published.load() + refused.load(), builders * rounds);
    ASSERT_EQ(verify_pack(dest, spec), 0U);
    ASSERT_EQ(entries(dir.path), std::vector<std::string>{spec.file_name()});
}
