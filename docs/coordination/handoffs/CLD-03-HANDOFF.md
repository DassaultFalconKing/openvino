# CLD-03-HANDOFF — OTD Trace Integrity and Offline Accounting (Cloud Wave 01)

## 1. Status and evidence class

Code authored and **unit-tested on Windows only**: the C++ trace writer through a standalone gtest harness, the Python tools with `unittest`. Not built inside `ov_gpu_unit_tests`, not compiled on Linux, no GPU involvement (the writer is host-only code and no inference was run). `ACCEPTED=NO`.

| Check | Status | Receipt |
|---|---|---|
| `python -m unittest test_otd_sim` (Python 3.14, Windows) | `UNIT_PASS` | 33/33, exit 0 (16 existing, 17 new) |
| Red run: the test file (before the CRLF test was added) against the **base** `otd_sim.py` | 14 of 16 new tests FAIL (expected); 16/16 existing pass | exit 1 (11 errors, 19 failures counting the 17 failing subtests of the rejection table separately; base rejects only "unknown call"); the two passing new tests need no new behavior (interleaved order, comment compatibility) |
| Base vs new `otd_sim.py` on a deterministic synthetic trace (`synth_trace.py --sessions 3 --spec-window 4 --seed 7`), comparing all pre-existing keys of `verify`, `summary`, `simulate --committed-tokens 100`, `simulate --capacity-fraction 0.5`, `simulate --legacy-partial-lease` | identical output, exit 0 for both | exact LRU replay semantics unchanged |
| C++ `moe_otd_trace.*` suite in the standalone harness | `UNIT_PASS` (standalone harness) | 13/13, exit 0 (9 existing, 4 new). On the base the new tests do not compile (`close()` does not exist) |
| Cross-language: a throwaway driver (not committed) linked against the real `moe_otd_trace.cpp` wrote a trace through explicit `close()` and through the destructor; `otd_sim.py verify` on both | exit 0, 0 mismatches, `complete=true`, 1 incomplete call | also verified with CRLF line endings (what a text-mode `ofstream` writes on Windows) |
| `/Zs` syntax check with the cohort build flags: `moe_otd_trace.cpp`, `expert_weight_providers.cpp` (includes the header), and the test TU with the real `test_utils.h` | `STATIC_CHECK_PASS` | exit 0 for all three (the C4244 warnings in `test_utils.h`/`xutility` and the C4996 `getenv` warning are pre-existing) |
| Linux GCC/Clang compile, Python on Linux | `BLOCKED_ENV` | no Linux toolchain on the author host |
| `ov_gpu_unit_tests` build and run | `NOT_RUN` | |
| Python older than 3.14 | `NOT_RUN` | no 3.10+-only syntax is used, but nothing else was run |

## 2. Source locks

- Repository `DassaultFalconKing/openvino`, base `research/ov-gpu-otd-cohort-b5bfb148` @ `479219245f6f10b670367ad35af16513d72d2091` (same pin as CLD-01/CLD-02).
- Branch `feature/otd-trace-accounting-cloud-20261022`; code commit `a973b0069aeafc52481240eb35534386c0781474`.
- **Independent of CLD-01 and CLD-02**: no file in common with either PR.
- Changed paths (mission limit: the four trace/simulator files, plus the tool README and this handoff):
  - `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_trace.hpp`
  - `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_trace.cpp`
  - `src/plugins/intel_gpu/tests/unit/test_cases/moe_offload_trace_test.cpp`
  - `src/plugins/intel_gpu/tools/moe_otd_sim/otd_sim.py`
  - `src/plugins/intel_gpu/tools/moe_otd_sim/test_otd_sim.py`
  - `src/plugins/intel_gpu/tools/moe_otd_sim/README.md`
  - `docs/coordination/handoffs/CLD-03-HANDOFF.md`
- Not changed: `expert_weight_providers.*` (the trace call sites), `synth_trace.py`, LRU, router, packer, OVMS integration `746dc833dc3bcccf64fabdb469feda9260582bae`, C6 `ba6ec792c3121850fea167fa7d7e190b25fcdc76`.

## 3. What changed / why / non-goals

### Gaps in the base

1. **Lost calls were invisible.** `TraceWriter` buffers a call until `end_call`. A load that throws (for example a CLD-01 checksum failure) skips `end_call`, so the call and the experts it already loaded vanished from the trace, with no sign anything was lost. A killed process lost its unflushed tail just as silently.
2. **The parser accepted inconsistent traces.** It did not check field counts beyond the minimum, duplicate ids, value ranges, or truncated last lines, so a damaged trace could produce plausible but wrong replay numbers.
3. **Bytes were ambiguous.** `bytes` in `E` records is the expert payload the runtime requested, not what the NVMe device served (page cache, read-ahead, filesystem amplification). Reports did not say so.
4. **Run facts had no schema.** Committed tokens came only from a CLI flag, with nothing tying them to the trace they describe.

### Writer (C++), backward compatible with v1

- New `TraceWriter::close()`, idempotent and also called by the destructor (a function-local static, so it runs at normal process exit). It writes every pending call as `# incomplete C,...` (status `incomplete`) followed by its recorded experts as `# incomplete E,...`, then `# end,calls=<n>,experts=<n>,incomplete_calls=<n>,incomplete_experts=<n>` with the counts actually emitted, then flushes.
- All new records are `#` comments, which v1 parsers already skip, so the format version stays `moe_otd_trace v1`. Stripping the comment lines leaves exactly the trace the base writer would have produced (tested).
- After `close()`, `register_layer`, `begin_call`, `expert` and `end_call` write nothing, so the end-record counts stay final.

### Simulator (Python)

- **Strict parsing** with line numbers: exact field counts (L=7, C=8, E=13); values within uint64 (no negatives except `evicted=-1`); unique layer and call ids; one header; `E` records only for known calls on the call's layer; no `E` records on overflowing calls; hits never evict; hit flag 0/1; `routed >= unique`; no duplicate unique experts; slots below capacity and experts below `num_expert`; nothing after the end record; end-record counts must match what was read, otherwise `records were lost`.
- **Truncation.** A final line without a terminator, when earlier lines have one, is treated as truncated even if it parses, because a cut inside the last number still looks valid. Strict mode fails. `--salvage` discards only that line and reports its number. A malformed record earlier in the file is never skipped.
- **Missing end record** (crash, kill, older writer, synthetic traces): parses with a warning and `trace_integrity.complete=false`.
- **Incomplete calls** are appended last in replay order. `verify` expects exactly their recorded prefix, and `replay` counts their recorded accesses.
- **Accounting.** `logical_expert_bytes` and `logical_reads` are reported explicitly. `physical_nvme_read_bytes` and `physical_read_counter_source` are `NOT_EXPOSED` unless an `otd_run_meta v1` sidecar supplies them. Simulated policy results always report physical bytes as `NOT_EXPOSED`, since a different policy changes physical traffic in ways the trace cannot predict.
- **Sidecar `otd_run_meta v1`** (`--run-meta`): optional fields trace_sha256, openvino/genai/ovms SHAs, runtime, quant, target/drafter model, workload C1/C2/C4, kv_cache_precision, context/proposed/accepted/committed tokens, parser/tool-loop result, and physical bytes with their counter source. Unknown keys, wrong types, bool-as-int, negatives, empty strings, `accepted > proposed`, physical bytes without a source, and a `trace_sha256` that differs from the trace file are errors. Absent fields print as `NOT_EXPOSED`; nothing is inferred.
- **CLI.** `--salvage` and `--format json|csv` on every command. Every report carries `trace_integrity`, which includes the SHA-256 of the trace bytes. Exit 2 means the trace or sidecar was rejected (message on stderr); `verify` still exits 1 on replay mismatches.

### Behavior changes reviewers must accept

- A trace whose last line lacks a terminator, which the base accepted, now fails unless `--salvage` is passed. The one existing test that built such a trace (`test_verify_detects_tampering`, via `"\n".join`) was given a trailing newline. Its assertion is unchanged.
- Parse errors used to escape as Python tracebacks (exit 1); they are now exit 2 with a one-line message.
- `logical_expert_bytes`, `logical_reads`, `physical_nvme_read_bytes`, `trace_integrity`, `run_meta` and `incomplete_calls` are new keys. No existing key changed value (section 1).

### Non-goals

Scheduler, prefetch, eviction policy, the trace call sites in `expert_weight_providers.cpp`, router weights, speculative positions, request identity, pack integrity (CLD-01), pack publication (CLD-02).

## 4. Invariants and limitations

- **T1:** a closed trace accounts for every call that began: emitted complete calls plus incomplete calls equals `begin_call` count, except calls begun after `close()`.
- **T2:** the parser either rejects a trace or every record it returns came from the file; salvage drops at most the final unterminated line, and says so.
- **T3:** no report contains a physical byte count, request, speculative or token figure that was not recorded by the runtime or supplied by a validated sidecar.
- **T4:** for traces the base parser accepted and the strict parser accepts, every pre-existing output key is unchanged.

`NOT_EXPOSED` (not in the trace, never fabricated): physical NVMe bytes and their source; router top-k weights; request/sequence identity; draft, verify, accepted and rejected positions; proposed, accepted and committed token counts; model, quant, KV precision and source SHAs; parser/tool-loop results. The sidecar can supply everything on this list except router weights, positions and request identity.

Limitations:

- **An incomplete call lacks the miss that threw.** `acquire_slot` writes the `E` record only after a successful fill, but the LRU lookup has already evicted and reassigned the slot, leaving it `filled=false`. The trace does not show that state change, so replay of later calls on that layer can diverge. `verify` then reports mismatches; it does not hide them. Recording the failed miss needs a change at the call site (`expert_weight_providers.cpp`). That file is outside this task and is edited by CLD-01, so it would be a follow-up based on CLD-01.
- Incomplete calls are replayed last, not at their real position; the warning says so.
- No end record after a crash, `abort`, `TerminateProcess` or `quick_exit`: the trace is reported `complete=false`, which is the intended signal.
- **Pre-existing, unchanged:** the writer is a function-local static. Code running from another static destructor after it is destroyed and calling `get_trace_writer()` would be undefined behavior. No such caller is known.
- Python was run on 3.14 only; Linux not run.

### Cost

Runtime: two counter increments under the existing mutex per `end_call`, plus one end record at exit. Writer output is unchanged for calls that complete. Offline parsing does more validation, which is still linear in the trace size.

## 5. Offline and unit tests

### Exact commands used by the author (Windows)

```bat
cd src\plugins\intel_gpu\tools\moe_otd_sim
python -m unittest -v test_otd_sim

rem C++ trace suite, standalone harness (shim test_utils.h = #pragma once + #include <gtest/gtest.h>)
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set GT=<openvino>\thirdparty\gtest\gtest\googletest
cl /nologo /std:c++17 /EHsc /MD /O1 /c /I%GT%\include /I%GT% %GT%\src\gtest-all.cc %GT%\src\gtest_main.cc
cl /nologo /std:c++17 /EHsc /MD /O1 /W4 /permissive- /Ishim\include /I%GT%\include /I<branch>\src\plugins\intel_gpu\src\graph\impls ^
   <branch>\src\plugins\intel_gpu\src\graph\impls\ocl_v2\moe\moe_otd_trace.cpp ^
   <branch>\src\plugins\intel_gpu\tests\unit\test_cases\moe_offload_trace_test.cpp ^
   gtest-all.obj gtest_main.obj /Fe:moe_otd_trace_tests.exe
moe_otd_trace_tests.exe
```

Semantic-preservation check: generate one synthetic trace, run base and branch `otd_sim.py` on it with the commands from section 1, and compare the JSON after removing the new keys listed in section 3.

### In the real test binary (NOT RUN)

```powershell
cmake --build <build-dir> --target ov_gpu_unit_tests --parallel 8
& <ov_gpu_unit_tests.exe> '--gtest_filter=moe_otd_trace.*'
```

### Linux (`BLOCKED_ENV` for the author)

```bash
g++ -std=c++17 -O1 -Wall -Wextra -pthread -Ishim/include -I$GT/include -I$GT -I<branch>/src/plugins/intel_gpu/src/graph/impls \
  $GT/src/gtest-all.cc $GT/src/gtest_main.cc \
  <branch>/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_trace.cpp \
  <branch>/src/plugins/intel_gpu/tests/unit/test_cases/moe_offload_trace_test.cpp -o trace_tests && ./trace_tests
cd src/plugins/intel_gpu/tools/moe_otd_sim && python3 -m unittest -v test_otd_sim
```

## 6. Windows 11 / Arc 140V local agent instructions

1. `git fetch origin feature/otd-trace-accounting-cloud-20261022`, check out detached, `git status --short` clean, `git diff 479219245f6f10b670367ad35af16513d72d2091 --stat` shows only the paths in section 2.
2. Run the Python suite and the standalone C++ harness (no GPU, seconds), then the real `ov_gpu_unit_tests` filter.
3. After the operator lifts the compute pause, on one existing short OTD run with `MOE_OTD_TRACE=<scratch file>`: confirm the file ends with `# end,...`, that `otd_sim.py verify` exits 0 with `complete=true`, and that `summary` reports `physical_nvme_read_bytes: NOT_EXPOSED`. Then kill a second run mid-decode and confirm `complete=false`, or a truncation error that `--salvage` reports.
4. Keep traces local. They contain only hashes and counts, but do not publish them without review. Do not change drivers or global environment variables.

## 7. Linux / P720 parity checks

- Run the Linux commands above with GCC and Clang, and the Python suite with the system Python.
- A Linux trace uses LF line endings; the CRLF path is covered by `test_crlf_trace_from_windows_text_stream_parses`.
- No GPU equivalence is implied by any Linux result.

## 8. Performance acceptance

No serving-path change worth measuring beyond noise: the writer is active only when `MOE_OTD_TRACE` is set. If measured, compare tokens/s of one traced run before and after on the same machine; any difference above run-to-run noise is a STOP.

## 9. Independent review assignment

- **Reviewer 1 (C++):** `close()` ordering under the mutex, idempotence, drop-after-close, destructor behavior, T1. Check that no existing record format changed byte-for-byte.
- **Reviewer 2 (build):** Windows MSVC and Linux GCC/Clang builds of the trace TU and test.
- **Reviewer 3 (offline tools/tests):** rerun the red, green and semantic-preservation checks; try to construct a damaged trace that the strict parser accepts with wrong numbers; review the `NOT_EXPOSED` list against the mission.
- **Reviewer 4:** traced-run overhead only (section 8), after authorization.
- Order: independent of CLD-01/CLD-02; can merge in any order.

**LOCAL-AGENT-BOOTSTRAP:** You are one reviewer above. Fetch only this branch and this file. Do not trust the author's results: rerun the Python suite, the C++ harness and the base-vs-branch comparison yourself (for the red run, put the base `otd_sim.py` next to this branch's `test_otd_sim.py`). Report commit SHA, exact command, exit code and any objection with file/line. Do not merge, do not touch C6, and do not run GPU workloads during the operator pause.

## 10. STOP / rollback / open questions

- Rollback: revert this branch's commit. Traces written by the new writer stay readable by the old parser (the new records are comments); traces from the old writer parse with a "no end record" warning.
- STOP if a trace accepted by the base parser and produced by the real runtime is rejected by the strict parser, or if any pre-existing output key changes value.
- Open questions for the operator: (a) record the failed miss at the call site in a follow-up based on CLD-01 (section 4)? (b) should `synth_trace.py` emit an end record so synthetic traces report `complete=true`? (c) which OS counter should be the reference `physical_read_counter_source` on Windows and on Linux?
