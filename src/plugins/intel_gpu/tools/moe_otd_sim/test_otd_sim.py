# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import io
import json
import os
import random
import sys
import tempfile
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
        result = verify_lru(parse_trace(io.StringIO("\n".join(lines) + "\n")))
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

    def test_runtime_never_loads_for_overflowing_leases(self):
        _, trace, _ = synth(capacity=8, sessions=3, prompt_tokens=16)
        fixed = replay(trace, "lru")["total"]
        legacy = replay(trace, "lru", legacy_partial_lease=True)["total"]
        self.assertEqual(fixed["overflow_partial_loads"], 0)
        self.assertGreater(legacy["overflow_partial_loads"], 0)
        self.assertEqual(fixed["overflow_calls"], legacy["overflow_calls"])
        self.assertLess(fixed["misses"], legacy["misses"])

    def test_overflowing_lease_has_no_expert_records(self):
        _, trace, _ = synth(capacity=8, sessions=3, prompt_tokens=16)
        overflowing = [c for c in trace.calls if c.kind == "S" and c.overflow]
        self.assertTrue(overflowing)
        self.assertTrue(all(not c.events for c in overflowing))

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


# Hand-checkable fixture: capacity 2. Call 0 misses expert 3 into slot 0; call 1 hits 3 and loads 4
# into slot 1. Matches what TraceWriter emits, including the trailing comma-separated expert list.
VALID = (
    "# moe_otd_trace v1\n"
    "L,0,aa,bb,2,8,1\n"
    "C,0,100,0,O,1,ok,3\n"
    "E,0,110,0,3,0,0,-1,4096,1,10,0,5\n"
    "C,1,200,0,S,2,ok,3 4\n"
    "E,1,210,0,3,0,1,-1,0,0,0,0,0\n"
    "E,1,220,0,4,1,0,-1,4096,1,10,0,5\n"
)
END = "# end,calls=2,experts=3,incomplete_calls=0,incomplete_experts=0\n"
# Expert 5 then misses, evicting the least recently used expert 3 from slot 0, and the load throws.
INCOMPLETE = (
    "# incomplete C,2,300,0,S,2,incomplete,5 6\n"
    "# incomplete E,2,310,0,5,0,0,3,4096,1,10,0,5\n"
)
END_WITH_INCOMPLETE = "# end,calls=2,experts=3,incomplete_calls=1,incomplete_experts=1\n"


def parse_text(text, **kwargs):
    return parse_trace(io.StringIO(text), **kwargs)


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        rc = otd_sim.main(args)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, out.getvalue(), err.getvalue()


class TraceIntegrityTest(unittest.TestCase):
    def test_v1_trace_without_end_record_still_parses_with_warning(self):
        trace = parse_text(VALID)
        integrity = trace.integrity.as_dict()
        self.assertFalse(integrity["complete"])
        self.assertFalse(integrity["end_record"])
        self.assertEqual(integrity["records"], 6)
        self.assertTrue(any("no end record" in w for w in integrity["warnings"]))
        result = verify_lru(trace)
        self.assertEqual((result["checked_events"], result["mismatches"]), (3, 0))

    def test_end_record_marks_trace_complete(self):
        trace = parse_text(VALID + END)
        self.assertTrue(trace.integrity.complete)
        self.assertEqual(trace.integrity.warnings, [])

    def test_end_record_count_mismatch_reports_lost_records(self):
        lines = (VALID + END).splitlines(keepends=True)
        del lines[5]  # drop one E record
        with self.assertRaisesRegex(TraceFormatError, "records were lost"):
            parse_trace(lines)

    def test_truncated_final_record_strict_fails_salvage_discards_only_it(self):
        truncated = VALID + "E,1,230,0,5,1"
        with self.assertRaisesRegex(TraceFormatError, "line 8: final record is truncated"):
            parse_text(truncated)
        trace = parse_text(truncated, salvage=True)
        self.assertEqual(trace.integrity.salvaged_line, 8)
        self.assertFalse(trace.integrity.complete)
        self.assertEqual(sum(len(c.events) for c in trace.calls), 3)

    def test_unterminated_final_line_is_treated_as_truncated_even_if_parseable(self):
        # A cut inside the last number can still look valid; only the missing terminator reveals it.
        cut = VALID.rstrip("\n")[:-1]
        with self.assertRaises(TraceFormatError):
            parse_text(cut)
        self.assertEqual(sum(len(c.events) for c in parse_text(cut, salvage=True).calls), 2)

    def test_malformed_record_before_the_end_is_never_salvaged(self):
        lines = VALID.splitlines(keepends=True)
        lines[3] = "E,0,110,0,3\n"
        for salvage in (False, True):
            with self.assertRaisesRegex(TraceFormatError, "line 4: malformed E record"):
                parse_trace(lines, salvage=salvage)

    def test_inconsistent_records_are_rejected(self):
        cases = {
            "duplicate layer": VALID + "L,0,cc,dd,2,8,1\n",
            "duplicate call id": VALID + "C,1,300,0,O,1,ok,5\n",
            "unknown call": VALID + "E,9,300,0,5,0,0,-1,0,0,0,0,0\n",
            "incomplete event of complete call": VALID + "# incomplete E,1,300,0,5,0,0,-1,0,0,0,0,0\n",
            "negative bytes": VALID + "C,2,300,0,O,1,ok,5\nE,2,310,0,5,0,0,3,-1,1,10,0,5\n",
            "uint64 overflow": VALID + "C,2,300,0,O,1,ok,5\nE,2,310,0,5,0,0,3,18446744073709551616,1,10,0,5\n",
            "slot beyond capacity": VALID + "C,2,300,0,O,1,ok,5\nE,2,310,0,5,2,0,3,4096,1,10,0,5\n",
            "expert beyond num_expert": VALID + "C,2,300,0,O,1,ok,8\n",
            "hit that evicts": VALID + "C,2,300,0,O,1,ok,4\nE,2,310,0,4,1,1,3,0,0,0,0,0\n",
            "bad hit flag": VALID + "C,2,300,0,O,1,ok,4\nE,2,310,0,4,1,2,-1,0,0,0,0,0\n",
            "overflow with events": VALID + "C,2,300,0,S,3,overflow,5 6 7\nE,2,310,0,5,0,0,3,4096,1,10,0,5\n",
            "routed below unique": VALID + "C,2,300,0,S,1,ok,5 6\n",
            "duplicate unique expert": VALID + "C,2,300,0,S,2,ok,5 5\n",
            "unknown kind": VALID + "C,2,300,0,X,1,ok,5\n",
            "event on other layer": VALID + "L,1,cc,bb,2,8,1\nC,2,300,0,O,1,ok,5\nE,2,310,1,5,0,0,3,4096,1,10,0,5\n",
            "record after end": VALID + END + "C,2,300,0,O,1,ok,5\n",
            "second header": VALID + "# moe_otd_trace v1\n",
            "bad end record": VALID + "# end,calls=2,experts=3\n",
        }
        for name, text in cases.items():
            with self.subTest(name):
                with self.assertRaises(TraceFormatError):
                    parse_text(text)

    def test_interleaved_calls_keep_completion_order_and_verify(self):
        text = (
            "# moe_otd_trace v1\n"
            "L,0,aa,bb,2,8,1\n"
            "L,1,cc,dd,2,8,1\n"
            "C,1,150,1,O,1,ok,2\n"           # begun after call 0, completed first
            "E,1,160,1,2,0,0,-1,4096,1,10,0,5\n"
            "C,0,100,0,O,1,ok,1\n"
            "E,0,170,0,1,0,0,-1,4096,1,10,0,5\n"
            "# end,calls=2,experts=2,incomplete_calls=0,incomplete_experts=0\n"
        )
        trace = parse_text(text)
        self.assertEqual([c.call for c in trace.calls], [1, 0])
        self.assertEqual(verify_lru(trace)["mismatches"], 0)

    def test_incomplete_call_is_replayed_with_its_recorded_prefix(self):
        trace = parse_text(VALID + INCOMPLETE + END_WITH_INCOMPLETE)
        self.assertTrue(trace.integrity.complete)  # file was closed
        self.assertFalse(trace.integrity.policy_evidence_eligible)  # calls did not all finish
        self.assertEqual((trace.integrity.incomplete_calls, trace.integrity.incomplete_events), (1, 1))
        self.assertTrue(trace.calls[-1].incomplete)
        self.assertTrue(any("never completed" in w for w in trace.integrity.warnings))
        result = verify_lru(trace)
        self.assertFalse(result["trace_integrity"]["policy_evidence_eligible"])
        self.assertEqual((result["checked_events"], result["mismatches"]), (4, 0))
        total = replay(trace, "lru")["total"]
        self.assertEqual((total["accesses"], total["misses"], total["evictions"]), (4, 3, 1))

    def test_incomplete_records_are_comments_for_v1_readers(self):
        for line in (INCOMPLETE + END_WITH_INCOMPLETE).splitlines():
            self.assertTrue(line.startswith("#"), line)
        # Without the extensions the remaining records are the plain v1 trace.
        stripped = "".join(l for l in (VALID + INCOMPLETE + END_WITH_INCOMPLETE).splitlines(keepends=True)
                           if not l.startswith("# incomplete") and not l.startswith("# end"))
        self.assertEqual(stripped, VALID)


class AccountingTest(unittest.TestCase):
    def test_summary_keeps_logical_bytes_and_never_invents_physical_bytes(self):
        summary = otd_sim.measured_summary(parse_text(VALID + END))
        self.assertEqual(summary["logical_expert_bytes"], 8192)
        self.assertEqual(summary["bytes"], 8192)
        self.assertEqual(summary["logical_reads"], 2)
        self.assertEqual(summary["physical_nvme_read_bytes"], otd_sim.NOT_EXPOSED)
        self.assertEqual(summary["physical_read_counter_source"], otd_sim.NOT_EXPOSED)

        meta = otd_sim.validate_run_meta({"schema": otd_sim.RUN_META_SCHEMA, "physical_nvme_read_bytes": 12288,
                                          "physical_read_counter_source": "os_disk_counter"})
        summary = otd_sim.measured_summary(parse_text(VALID + END), meta)
        self.assertEqual(summary["physical_nvme_read_bytes"], 12288)
        self.assertEqual(summary["logical_expert_bytes"], 8192)

    def test_simulation_never_reports_physical_bytes(self):
        result = replay(parse_text(VALID + END), "clock", capacity=1)
        self.assertEqual(result["physical_nvme_read_bytes"], otd_sim.NOT_EXPOSED)
        self.assertEqual(result["total"]["logical_expert_bytes"], result["total"]["bytes"])

    def test_run_meta_validation(self):
        schema = {"schema": otd_sim.RUN_META_SCHEMA}
        full = otd_sim.validate_run_meta(dict(schema, workload="C2", proposed_tokens=30, accepted_tokens=20, committed_tokens=26))
        self.assertIsNone(full["drafter_model"])
        self.assertEqual(otd_sim._report_meta(full)["drafter_model"], otd_sim.NOT_EXPOSED)
        bad = {
            "missing schema": {},
            "other schema": {"schema": "otd_run_meta v2"},
            "unknown field": dict(schema, speculative_acceptance=0.9),
            "bad workload": dict(schema, workload="C3"),
            "accepted over proposed": dict(schema, proposed_tokens=3, accepted_tokens=4),
            "physical without source": dict(schema, physical_nvme_read_bytes=10),
            "bool as int": dict(schema, committed_tokens=True),
            "negative": dict(schema, context_tokens=-1),
            "empty string": dict(schema, quant=""),
        }
        for name, meta in bad.items():
            with self.subTest(name):
                with self.assertRaises(otd_sim.RunMetaError):
                    otd_sim.validate_run_meta(meta)


class CliIntegrityTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)

    def _write(self, name, text):
        path = os.path.join(self._dir.name, name)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        return path

    def test_truncated_file_fails_without_salvage_and_reports_with_it(self):
        path = self._write("t.trace", VALID + "E,1,230,0,5")
        rc, _, err = run_cli(["summary", path])
        self.assertEqual(rc, 2)
        self.assertIn("truncated", err)
        rc, out, _ = run_cli(["summary", path, "--salvage"])
        self.assertEqual(rc, 0)
        integrity = json.loads(out)["trace_integrity"]
        self.assertEqual(integrity["salvaged_final_record_line"], 8)
        with open(path, "rb") as fh:
            self.assertEqual(integrity["source_sha256"], hashlib.sha256(fh.read()).hexdigest())

    def test_run_meta_must_match_trace_and_supplies_committed_tokens(self):
        trace_path = self._write("t.trace", VALID + END)
        digest = hashlib.sha256((VALID + END).encode("utf-8")).hexdigest()
        wrong = self._write("wrong.json", json.dumps({"schema": otd_sim.RUN_META_SCHEMA, "trace_sha256": "0" * 64}))
        self.assertEqual(run_cli(["summary", trace_path, "--run-meta", wrong])[0], 2)

        meta = self._write("meta.json", json.dumps({"schema": otd_sim.RUN_META_SCHEMA, "trace_sha256": digest,
                                                    "committed_tokens": 4}))
        rc, out, _ = run_cli(["simulate", trace_path, "--policy", "lru", "--run-meta", meta])
        self.assertEqual(rc, 0)
        total = json.loads(out)[0]["total"]
        self.assertEqual(total["logical_expert_bytes_per_committed_token"], 8192 / 4)
        self.assertEqual(json.loads(out)[0]["run_meta"]["accepted_tokens"], otd_sim.NOT_EXPOSED)
        self.assertTrue(json.loads(out)[0]["trace_integrity"]["policy_evidence_eligible"])
        rc, _, err = run_cli(["simulate", trace_path, "--run-meta", meta, "--committed-tokens", "5"])
        self.assertEqual(rc, 2)
        self.assertIn("differs", err)

    def test_crlf_trace_from_windows_text_stream_parses(self):
        # The runtime opens the trace as a text-mode ofstream, so Windows traces end lines with CRLF.
        path = self._write("t.trace", (VALID + END).replace("\n", "\r\n"))
        rc, out, _ = run_cli(["verify", path])
        self.assertEqual(rc, 0)
        self.assertTrue(json.loads(out)["trace_integrity"]["complete"])
        path = os.path.join(self._dir.name, "bad.trace")
        with open(path, "wb") as fh:
            fh.write(b"# moe_otd_trace v1\n\xff\n")
        self.assertEqual(run_cli(["verify", path])[0], 2)

    def test_csv_output_lists_not_exposed_metrics(self):
        path = self._write("t.trace", VALID + END)
        rc, out, _ = run_cli(["summary", path, "--format", "csv"])
        self.assertEqual(rc, 0)
        rows = list(csv.reader(io.StringIO(out)))
        self.assertEqual(rows[0], ["result", "metric", "value"])
        self.assertIn(["0", "physical_nvme_read_bytes", "NOT_EXPOSED"], rows)
        self.assertIn(["0", "logical_expert_bytes", "8192"], rows)
        self.assertIn(["0", "trace_integrity.complete", "True"], rows)


if __name__ == "__main__":
    unittest.main()
