// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "../../src/graph/impls/ocl_v2/moe/moe_otd_execution_guard.hpp"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using ov::intel_gpu::ocl::moe_otd::SynchronousExecutionGate;
static void check(bool ok) { if (!ok) throw std::runtime_error("execution gate regression"); }
int main(int argc, char** argv) {
    if (argc == 2) {
        const std::string mode(argv[1]);
        ov::intel_gpu::ocl::moe_otd::drain_or_fail_stop(
            [&] {
                std::fputs("ONEDNN_DRAIN_ATTEMPTED\n", stderr);
                if (mode == "--failed-drain" || mode == "--both-failed") throw std::runtime_error("oneDNN wait failed");
            },
            [&] {
                std::fputs("OCL_DRAIN_ATTEMPTED\n", stderr);
                if (mode == "--failed-ocl-drain" || mode == "--both-failed") throw std::runtime_error("OCL finish failed");
            });
        return mode == "--good-drain" ? 0 : 1;
    }
    SynchronousExecutionGate gate;
    {
        auto resident_scope = gate.enter([] { throw std::runtime_error("r0 must bypass drain"); }, false);
    }
    std::atomic<int> in_flight{0}, drains{0};
    std::vector<std::thread> requests;
    for (int i = 0; i < 4; ++i) {
        requests.emplace_back([&] {
            for (int iteration = 0; iteration < 100; ++iteration) {
                auto scope = gate.enter([&] {
                    check(in_flight.fetch_sub(1) == 1);
                    ++drains;
                });
                check(in_flight.fetch_add(1) == 0);
            }
        });
    }
    for (auto& request : requests) request.join();
    check(drains == 400 && in_flight == 0);
    bool caught = false;
    try {
        auto scope = gate.enter([&] { ++drains; });
        throw std::runtime_error("request cancelled");
    } catch (const std::runtime_error&) { caught = true; }
    check(caught && drains == 401);
    caught = false;
    try {
        auto scope = gate.enter([] { throw std::runtime_error("GPU drain failure"); });
    } catch (const std::runtime_error&) { caught = true; }
    check(caught);
    caught = false;
    try { auto scope = gate.enter([] {}); }
    catch (const std::logic_error&) { caught = true; }
    check(caught);
    SynchronousExecutionGate unwind_gate;
    caught = false;
    try {
        auto scope = unwind_gate.enter([] { throw std::runtime_error("drain failed while cancelled"); });
        throw std::runtime_error("original cancellation");
    } catch (const std::runtime_error& e) { caught = std::string(e.what()) == "original cancellation"; }
    check(caught);
    caught = false;
    try { auto scope = unwind_gate.enter([] {}); }
    catch (const std::logic_error&) { caught = true; }
    check(caught);
    std::cout << "Execution gate: serialization, cancellation, poison and unwind PASS\n";
}
