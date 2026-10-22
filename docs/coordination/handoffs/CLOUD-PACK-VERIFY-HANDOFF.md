# CLOUD-PACK-VERIFY-HANDOFF — Independent read-only pack v1 inspector

**Evidence class:** `UNIT_PASS (cloud Python only)`, 14/14 deterministic tests, Python stdlib; **C++/GPU NOT_RUN**, `ACCEPTED=NO`. Do not conflate this independent v1 fixture suite with `ov_gpu_unit_tests` or Arc 140V serving acceptance.

## Source lock and ownership

Repo `DassaultFalconKing/openvino`; clean independent branch based on exact `research/ov-gpu-otd-cohort-b5bfb148` @ `479219245f6f10b670367ad35af16513d72d2091`. Owning PR and code HEAD: inspect the Draft PR link. Related cloud PRs #1/#4 (serving integrity/identity), #2 (atomic publication), #3/#5 (trace); this utility modifies **none** of their C++ code or trace tools. Source files: `src/plugins/intel_gpu/tools/moe_otd_pack/verify_pack.py`, `test_verify_pack.py`, `README.md` plus this document.

## Scope and correctness contract

- Offline **single-open-handle**, **read-only**, Python stdlib verifier for packed little-endian v1.
- Validate magic, version, canonical offsets/alignment, FNV header, file length, layer identity when expected key is supplied, expert-count bound, complete placement permutation, expert-id-indexed FNV table and every logical-expert **unpadded** payload.
- SHA-less, non-cryptographic checksums are corruption checks, not trust/authentication. File naming can be forged. A trusted model key must come from existing model/source metadata, not be copied from a suspicious header.
- Fixed 1MiB streaming chunks per record; `O(expert_count)` metadata and bad-ID state. **Adds zero serving reads / GPU allocations / runtime dependencies**.
- No mutation or pack publication, no OS device I/O measurements, no new scheduler or DRAM cache layer. Concurrent in-place writes remain unsafe; evaluate scratch copies while builders are idle.

## Actually executed tests

Command (cloud Python on synthetic files): `cd /mnt/data/otd_pack_verifier_work && python -m unittest -v test_verify_pack`.
Result: `Ran 14 tests ... OK` (exit 0), including valid permuted placement, first/last data corruption, checksum corruption, duplicate placement, resealed noncanonical header, truncated file, wrong model ID, renamed file, expert allocation limit, >2MiB streaming and CLI exit statuses.

**NOT_RUN:** Windows PowerShell/Python, native C++ `build_pack` differential, actual OTD pack in serving folder, Intel GPU unit/plugin tests, Arc 140V correctness, r40/r75 profiling. Linux machine is unavailable and `LINUX_DEFERRED_NO_MACHINE` is a scope limitation, not a pass.

## Windows local QA sequence

1. In a clean separate worktree, check out the exact inspector Draft PR branch and verify HEAD, status, scope and Python version.
2. Run `python -m unittest -v test_verify_pack`; capture exit, summary and SHA. No GPU needed.
3. **On a scratch pack from the actual combined OpenVINO `build_pack()`**, pass `--expect-key` and `--expect-experts`, compare with native `verify_pack` and `PackLayout::open`. Check logical expert checksum index under a non-identity placement.
4. Copy into separate scratch files; corrupt each of: payload first/last, checksum table, placement, header with recomputed FNV, truncation. Expect native and Python consensus where contracts overlap.
5. Review source independently for v1/struct agreement, under/overflow, Python 3.10+ syntax, long-running FNV cost (test offline wall only); no slow full pack scan on production inference workers.
6. Report results as `PASS|FAIL|NOT_RUN|BLOCKED` with commands, SHA, failures, examples and device/scratch path redaction. Run GPU benchmarks **only** if the operator separately authorizes it.

## Integration and rollback

This PR remains Draft until the existing Windows integration agent finishes its mandatory build, negative/provider tests and explicit GPU gates. The coordinator can then add this independent file-only commit to the approved Windows integration branch, run the Python suite and native differential test, and record it in acceptance docs. Do not merge just because 14 synthetic Python tests pass. Rollback: remove these three tool files plus this handoff; the runtime/pack format remain untouched.

**LOCAL-AGENT-BOOTSTRAP:** You are the independent Windows pack-inspector reviewer. Fetch this PR **without changing the main integration branch**; verify its HEAD, run `python -m unittest -v test_verify_pack` and compare the CLI to native `build_pack`/`verify_pack` on **copies** in a scratch folder. Record exact commands/exit codes, corruption test matrix and mismatches in your own evidence. Do not change C6, OVMS, production packs or driver and do not claim GPU acceptance. Report `LINUX_DEFERRED_NO_MACHINE` without trying to provision Linux.
