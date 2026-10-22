# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import otd_sim  # noqa: E402
from otd_sim import POLICIES, Belady, Clock, ExactLRU, TraceFormatError, TwoQ, parse_trace, replay, verify_lru  # noqa: E402
from synth_trace import SynthConfig, generate  # noqa: E402


def synth(**overrides):
    cfg = SynthConfig(**overrides)
    buf = io.StringIO()
    meta = generate(cfg, buf)
    text = buf.getvalue()
    return text, parse_trace(io.StringIO(text)), meta


class ExactLruTest(unittest.TestCase):
    def test_matches_runtime_refresh_order(self):
        # Mirrors moe_lru_cache.lru_order_refresh_on_access in the C++ unit tests.
        lru = ExactLRU(3)
        for e in (0, 1, 2):
            lru.access(e, set())
        lru.access(0, set())
        self.assertEqual(lru.access(3, set()), (False, 1, 1))

    def test_sequential_slots_then_recycle(self):
        lru = ExactLRU(2)
        self.assertEqual(lru.access(10, set()), (False, 0, None))
        self.assertEqual(lru.access(11, set()), (False, 1, None))
        self.assertEqual(lru.access(10, set()), (True, 0, None))
        self.assertEqual(lru.access(12, set()), (False, 1, 11))


class PolicyInvariantTest(unittest.TestCase):
    def _random_calls(self, seed, n_calls=400, num_expert=40, max_unique=6):
        rng = random.Random(seed)
        return [rng.sample(range(num_expert), rng.randint(1, max_unique)) for _ in range(n_calls)]

    def _run(self, policy, calls):
        resident_sizes = []
        misses = 0
        for experts in calls:
            pinned = set()
            for e in experts:
                hit, slot, evicted = policy.access(e, pinned)
                self.assertNotIn(evicted, pinned, "evicted an expert leased by the current call")
                self.assertIn(e, policy.resident())
                self.assertLess(slot, policy.capacity)
                pinned.add(e)
                misses += 0 if hit else 1
            resident_sizes.append(len(policy.resident()))
        self.assertLessEqual(max(resident_sizes), policy.capacity)
        return misses

    def test_policies_respect_pins_and_capacity(self):
        calls = self._random_calls(7)
        future = [e for c in calls for e in c]
        results = {}
        for name, factory in (("lru", ExactLRU), ("clock", Clock), ("2q", TwoQ)):
            results[name] = self._run(factory(8), calls)
        results["opt"] = self._run(Belady(8, future), calls)
        for name in ("lru", "clock", "2q"):
            self.assertLessEqual(results["opt"], results[name], name)

    def test_slots_are_unique_among_residents(self):
        calls = self._random_calls(11)
        for factory in (Clock, TwoQ):
            policy = factory(6)
            slot_of = {}
            for experts in calls:
                pinned = set()
                for e in experts:
                    hit, slot, evicted = policy.access(e, pinned)
                    pinned.add(e)
                    if evicted is not None:
                        slot_of.pop(evicted, None)
                    slot_of[e] = slot
                resident = policy.resident()
                slots = [slot_of[e] for e in resident]
                self.assertEqual(len(slots), len(set(slots)), factory.name)


class TraceRoundTripTest(unittest.TestCase):
    def test_plain_decode_trace_verifies(self):
        _, trace, _ = synth()
        result = verify_lru(trace)
        self.assertGreater(result["checked_events"], 0)
        self.assertEqual(result["mismatches"], 0, result["first_mismatches"])

    def test_overflow_and_fallback_trace_verifies(self):
        _, trace, _ = synth(capacity=8, sessions=3, prompt_tokens=16)
        self.assertTrue(any(c.overflow for c in trace.calls))
        self.assertTrue(any(c.fallback_of is not None for c in trace.calls))
        result = verify_lru(trace)
        self.assertEqual(result["mismatches"], 0, result["first_mismatches"])

    def test_replay_at_recorded_capacity_matches_recorded_counts(self):
        for overrides in ({}, {"capacity": 8, "sessions": 3}, {"spec_window": 4}):
            _, trace, _ = synth(**overrides)
            summary = otd_sim.measured_summary(trace)
            sim = replay(trace, "lru")["total"]
            self.assertEqual(sim["hits"], summary["hits"], overrides)
            self.assertEqual(sim["misses"], summary["loads"], overrides)
            self.assertEqual(sim["bytes"], summary["bytes"], overrides)

    def test_verify_detects_tampering(self):
        text, _, _ = synth()
        lines = text.splitlines()
        idx = next(i for i, l in enumerate(lines) if l.startswith("E,") and ",1,-1," in l)
        parts = lines[idx].split(",")
        parts[6] = "0"  # flip a recorded hit into a miss
        lines[idx] = ",".join(parts)
        result = verify_lru(parse_trace(io.StringIO("\n".join(lines))))
        self.assertGreater(result["mismatches"], 0)

    def test_parser_rejects_missing_header_and_orphan_events(self):
        with self.assertRaises(TraceFormatError):
            parse_trace(io.StringIO("L,0,a,b,1,2,1\n"))
        with self.assertRaises(TraceFormatError):
            parse_trace(io.StringIO("# moe_otd_trace v1\nL,0,a,b,1,2,1\nE,5,0,0,1,0,0,-1,0,0,0,0,0\n"))


class ReplayTest(unittest.TestCase):
    def test_bigger_capacity_skips_recorded_fallback(self):
        _, trace, _ = synth(capacity=8, sessions=3, prompt_tokens=16)
        small = replay(trace, "lru")["total"]
        big = replay(trace, "lru", capacity=128)["total"]
        self.assertGreater(small["overflow_calls"], 0)
        self.assertEqual(big["overflow_calls"], 0)
        self.assertLess(big["misses"], small["misses"])

    def test_smaller_capacity_synthesizes_fallback(self):
        _, trace, _ = synth(capacity=32)
        tiny = replay(trace, "lru", capacity=4)
        self.assertGreater(tiny["total"]["overflow_calls"], 0)
        self.assertIn("fallback", tiny["by_phase"])

    def test_precheck_never_loads_for_overflowing_leases(self):
        _, trace, _ = synth(capacity=8, sessions=3, prompt_tokens=16)
        base = replay(trace, "lru")["total"]
        pre = replay(trace, "lru", precheck=True)["total"]
        self.assertGreater(base["overflow_partial_loads"], 0)
        self.assertEqual(pre["overflow_partial_loads"], 0)
        self.assertEqual(pre["overflow_calls"], base["overflow_calls"])

    def test_opt_is_lower_bound_and_compulsory_misses_hold(self):
        _, trace, _ = synth(capacity=16, sessions=2, spec_window=3)
        compulsory = sum(len({e for c in trace.calls if c.layer == lid for e in c.unique}) for lid in trace.layers)
        results = {p: replay(trace, p)["total"]["misses"] for p in POLICIES}
        for policy, misses in results.items():
            self.assertGreaterEqual(misses, compulsory, policy)
            self.assertLessEqual(results["opt"], misses, policy)

    def test_speculative_windows_land_in_multi_token_phase(self):
        _, trace, meta = synth(spec_window=3)
        result = replay(trace, "lru")
        self.assertIn("multi_2to8tok", result["by_phase"])
        self.assertNotIn("decode_1tok", result["by_phase"])
        self.assertGreater(meta["rejected_tokens"], 0)

    def test_cli_simulate_reports_per_committed_token(self):
        text, _, meta = synth()
        path = os.path.join(os.environ.get("TMPDIR", os.environ.get("TEMP", ".")), "otd_sim_cli_test.trace")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            rc = otd_sim.main(["simulate", path, "--policy", "lru", "--committed-tokens", str(meta["committed_tokens"])])
        finally:
            sys.stdout = old
            os.remove(path)
        self.assertEqual(rc, 0)
        self.assertIn("bytes_per_committed_token", out.getvalue())


if __name__ == "__main__":
    unittest.main()
