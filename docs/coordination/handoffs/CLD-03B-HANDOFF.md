# CLD-03B-HANDOFF — Trace policy-evidence gate

**Status:** `IMPLEMENTED_NOT_RUN`. Draft follow-up based on CLD-03 PR #3; `ACCEPTED=NO`.

## Source and scope

- Repository `DassaultFalconKing/openvino`.
- Parent CLD-03 HEAD: `4e8fa732151e1169aa58faa9a5732f2b021d5b16`; original source `479219245f6f10b670367ad35af16513d72d2091`.
- This branch: `fix/otd-trace-policy-evidence-gate-20261022`, first commit `ebb7c805c797f9427d63654fcf645abe364b589c`; final HEAD inspect on PR.
- Only `src/plugins/intel_gpu/tools/moe_otd_sim/otd_sim.py`, `test_otd_sim.py`, `README.md` and this handoff. No OpenVINO runtime, trace producer, GPU plugin, OVMS or C6 source changes.
- This is **not** CLD-04: no scheduler, new policy, prefetch, expert hot-set, MTP or KV modification.

## Problem and algorithm

Original `TraceIntegrity.complete` correctly meant the **file** had a valid closing end record, but could be `True` even when `incomplete_calls>0`. This was easy to mistake for trustworthy routing evidence, although an aborted cache lookup/pack read could mutate LRU state without a complete `E` event. Incomplete calls are replayed last for diagnostics, which does not reconstruct original cache history.

New additive `trace_integrity.policy_evidence_eligible` is `True` only if:
1. The writer closed and an end record is present.
2. No line was salvaged from truncation.
3. No incomplete calls or incomplete expert events were recorded.

This is a **necessary, not sufficient**, condition for a real policy decision: reviewers must additionally obtain zero LRU mismatches on the full actual model trace and a trustworthy OS physical-NVMe counter and performance baseline. If the flag is false, `verify`, `summary`, `simulate` outputs are diagnostics, not evidence to promote a cache policy.

Legacy `trace_integrity.complete` and existing replay/verify calculations and CLI exit codes remain **unchanged** for compatibility. Additional warning is produced when policy evidence is ineligible. No speculative acceptance or request identity is fabricated.

## Negative tests added

- v1 trace without end record: parses with warning, policy gate `False`.
- Clean end record and all calls completed: policy gate `True`.
- Truncated/salvaged tail: policy gate `False`.
- File closed with incomplete calls, even if LRU replay has zero mismatches: policy gate `False`.
- Summary/simulate output carries gate via `trace_integrity`; existing source keys and original LRU replay semantics remain unchanged.

## Local QA commands (not run by cloud author)

```powershell
git fetch origin fix/otd-trace-policy-evidence-gate-20261022
git switch --detach origin/fix/otd-trace-policy-evidence-gate-20261022
cd src/plugins/intel_gpu/tools/moe_otd_sim
python -m unittest -v test_otd_sim
if ($LASTEXITCODE -ne 0) { throw "OTD offline tests failed" }
```

Linux:
```bash
git fetch origin fix/otd-trace-policy-evidence-gate-20261022
git switch --detach origin/fix/otd-trace-policy-evidence-gate-20261022
cd src/plugins/intel_gpu/tools/moe_otd_sim
python3 -m unittest -v test_otd_sim
```

For live acceptance (only with explicit operator permission), generate an actual complete trace with `MOE_OTD_TRACE`, run `otd_sim.py verify <trace>`, confirm zero mismatches and `policy_evidence_eligible=true`, and obtain measured OS physical NVMe bytes separately. A failed expert miss remains non-replayable in this wave; capture precise failure from CLD-01 provider and propose a separate, source-bound trace event if required.

## Independent reviews

Reviewer A: Python/backwards compatibility, `complete` semantics and baseline output keys; reviewer B: replay correctness on synthetic aborted traces (must **not** promote); reviewer C: Windows and Linux Python 3.10+/3.14 compatibility and error codes; reviewer D: after operator permission, real LRU trace and OS physical bytes. Report SHA, exact commands/exits and regression fixtures, not inherited claims.

**LOCAL-AGENT-BOOTSTRAP:** Fetch only the dependent Draft PR and this handoff, verify parent/changed paths, rerun Python unit suite independently, inspect fields in JSON and CSV for clean, incomplete, truncated and legacy-v1 trace files. Treat non-eligible traces as diagnostics. Do not change production schedules or run GPU workloads without explicit authorization.

## STOP / rollback

Stop if v1 parsing or exact LRU replay changes, or if an incomplete file is labelled eligible. Revert only this add-on commit; do not merge CLD-03B without CLD-03. Integration: CLD-03/#3 then this PR, independent of CLD-01/02 source changes.
