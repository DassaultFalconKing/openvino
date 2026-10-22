# CLD-02-HANDOFF — Atomic Pack Publication (Cloud Wave 01)

## 1. Status and evidence class

Code authored and **unit-tested on Windows only through a standalone harness**. Not built inside `ov_gpu_unit_tests`, not compiled on Linux, no GPU involvement (this patch has no GPU code). `ACCEPTED=NO`.

| Check | Status | Receipt |
|---|---|---|
| Red run of the new tests against the **unmodified** `build_pack` (Windows, MSVC 14.44) | 4 of 5 FAIL (expected) | exit 1; failures listed in section 3 |
| Green run of the new tests with this patch | `UNIT_PASS` (standalone harness) | 5/5, exit 0 |
| Same suite `--gtest_repeat=50` | `UNIT_PASS` (standalone harness) | exit 0 |
| Existing `moe_otd_pack.*` suite (11 tests) on this branch | `UNIT_PASS` (standalone harness) | 11/11, exit 0 |
| `/Zs` syntax check of `moe_otd_pack.cpp` with the exact cohort build flags, and of the new test TU with the real `test_utils.h` | `STATIC_CHECK_PASS` | exit 0 (the C4244 warnings in `test_utils.h`/`xutility` also occur on unmodified files) |
| Linux GCC/Clang compile of the POSIX branch (`fsync`, `rename(2)`) | `BLOCKED_ENV` | no Linux toolchain on the author host |
| `ov_gpu_unit_tests` build and run | `NOT_RUN` | |

**Standalone harness:** `moe_otd_pack.cpp` depends only on the C++ standard library, and the pack tests only use gtest macros. The harness compiles those two files plus `gtest-all.cc`/`gtest_main.cc` from `thirdparty/gtest`, with a one-line `test_utils/test_utils.h` that includes `<gtest/gtest.h>`. Recipe in section 5.

## 2. Source locks

- Repository `DassaultFalconKing/openvino`, base `research/ov-gpu-otd-cohort-b5bfb148` @ `479219245f6f10b670367ad35af16513d72d2091` (pinned, same as CLD-01).
- Branch `feature/otd-pack-atomic-publish-cloud-20261022`. **Independent of CLD-01** (PR #1): it does not depend on CLD-01 code. Expected textual overlap: both PRs edit `moe_otd_pack.cpp`, in different functions (CLD-01: `plan`/`open`/`verify_pack`/includes; CLD-02: `build_pack`, the start of `open`, a new include block). Reviewer 2 should do a trial merge of both onto the base.
- Changed paths:
  - `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_pack.cpp`
  - `src/plugins/intel_gpu/tests/unit/test_cases/moe_offload_pack_publish_test.cpp` (new)
  - `src/plugins/intel_gpu/tests/unit/sources.cmake` (one line)
  - `docs/coordination/handoffs/CLD-02-HANDOFF.md`
- OVMS integration `746dc833dc3bcccf64fabdb469feda9260582bae` and C6 `ba6ec792c3121850fea167fa7d7e190b25fcdc76`: untouched.

## 3. What changed / why / non-goals

### Defects in the base sequence (`remove(dest)` then `rename(tmp, dest)`)

These failures were **reproduced by tests on Windows**, not inferred from reading the code:

1. **Empty directory at the destination is deleted.** `remove()` deletes it and the pack is published in its place (`directory_at_destination_is_left_alone_and_fails`: base "throws nothing").
2. **A corrupt pack is reported as reused.** When a reader holds the destination open (the runtime opens packs with `FILE_SHARE_READ | FILE_SHARE_WRITE`, without `FILE_SHARE_DELETE`), `remove()` fails silently and `rename()` fails. The fallback then checks only `PackLayout::open` (the header) and returns `reused_existing=true` for the corrupt file that caused the rebuild (`corrupt_destination_held_open_is_never_reported_as_reused`: base `reused_existing == true`).
3. **Wrong placement returned as success.** Same path: a rebuild that asked for a new placement "succeeds" while the old placement stays on disk (`failed_replacement_keeps_previous_pack_and_never_returns_wrong_placement`).
4. **Readers fail during rebuilds.** Under concurrent rebuilds a reader failed 16+ opens in 30 ms (`concurrent_rebuilds_never_hide_the_destination_from_readers`). The base reported these as `truncated header`, because `PackLayout::open` never checked whether its `ifstream` actually opened.

On POSIX, defect 1 also applies. Defects 2 and 3 do not, because `rename(2)` succeeds over an open file. The remove-then-rename gap (a reader seeing "missing") exists on both, but was **not** separately reproduced on Linux.

### New sequence

1. Refuse to start if the random temp name already exists, checked with `symlink_status` so a planted link is never followed.
2. Write the temp file in the destination directory (same volume, so rename is a metadata operation).
3. `sync_to_disk(tmp)`: `FlushFileBuffers` on Windows, `fsync` (with `EINTR` retry) on POSIX, **before** publishing. Otherwise a crash after the rename could leave the name pointing at data that never reached the disk. The directory entry is not fsynced: losing the rename after a crash leaves the previous pack or no pack, both of which are safe.
4. **Single `std::filesystem::rename(tmp, dest)` with no prior `remove`.** POSIX `rename(2)` atomically replaces. MSVC implements it with `MoveFileExW(MOVEFILE_REPLACE_EXISTING)`, which replaces when nobody holds the destination open and fails otherwise. `rename` replaces a symlink or reparse point at the destination itself; it never writes through it.
5. **On rename failure:** delete the temp file, then accept the destination **only if** it opens, has the requested placement (when one was requested) and passes the full `verify_pack`. Otherwise throw `OTD pack: cannot publish <file name>: <OS error>; destination left unchanged (<reason>)`. The previous file is never modified.
6. `PackLayout::open` reports `cannot open` when the file exists but cannot be opened, instead of the misleading `truncated header`.

### Platform contract (state this exactly in review)

- **POSIX:** readers see the old or the new complete file at all times; a replace succeeds even while readers hold the old file (they keep the old inode).
- **Windows:** a replace is refused while any process holds the destination open without `FILE_SHARE_DELETE`, which includes every OpenVINO runtime reader. The builder then fails closed or reuses a fully verified identical pack. While the replace is in progress, a concurrent open can briefly fail with `cannot open` (2 transient failures were seen in one 30 ms stress run, 0 in the last of 50 repeats; distribution not measured). The reader **never** observed a missing, truncated or mixed file after the fix. Under `require` that transient failure is a loud configure-time error, not silent fallback. **No retry/sleep was added** to `PackLayout::open`: do that only as a deliberate product decision.
- Windows `ReplaceFileW` was considered and not used: it does not help while readers hold the file open, and it has more failure modes that partially apply (`ERROR_UNABLE_TO_MOVE_REPLACEMENT*`).

### Non-goals

Packer format, record layout, LRU, router, checksums (CLD-01), trace (CLD-03), scheduler, quant, KV U4, OVMS, C6.

## 4. Threat model and invariants

- **Invariant P1:** at no point does the destination name refer to a partially written file.
- **Invariant P2:** `build_pack` returns only if the destination now holds a complete, verified pack for `spec` with the requested placement (if one was given).
- **Invariant P3:** if `build_pack` throws after building, the destination bytes are unchanged and no `*.tmp-*` file from this call remains.
- **Invariant P4:** the destination is never deleted, and a non-file at the destination is never removed.
- **Not covered:** a crash or kill mid-build leaves a stale `*.tmp-*` file. It is never published, but it occupies disk space (up to one layer pack). Automatic cleanup is not implemented because it is unsafe against live concurrent builders without a lock. Operators can delete `otd-*.pack.tmp-*` while no builder runs.
- **Not covered (follow-up, depends on CLD-01):** identity between the file whose header/checksums `configure_pack` loaded and the file `ParallelWeightReader` later opens lazily on the first miss. On POSIX a rebuild in between makes the reader see the new inode. CLD-01's per-miss checksum turns a mismatched record into a fail-closed error, but valid-but-different content (for example another placement) would also be rejected. The fix is to open the reader eagerly and compare the header read through that handle. That edits `expert_weight_providers.cpp`, which CLD-01 also edits, so it belongs in a separate PR **based on CLD-01**.
- The temp-name check is check-then-open (TOCTOU). The name has 64 random bits and lives in an operator-configured directory; exclusive create (`O_EXCL`/`CREATE_NEW`) would close the gap but needs replacing `std::ofstream` with native handles. Not done.
- Thread safety: `build_pack` has no shared state; concurrent builders are coordinated only through the filesystem.

### Cost

- One extra `fsync`/`FlushFileBuffers` per built layer: build mode only, it waits for data the OS would write anyway. Serving cost: **none**.
- Failure path only: one full `verify_pack` of the destination (reads the whole pack, O(pack bytes)). That is the same cost as the existing reuse check at the start of `build_pack`.
- No new allocations or locks; no new dependency (`<windows.h>`/`<unistd.h>` are system headers).

## 5. Offline and unit tests

### Exact commands used by the author (Windows, standalone harness)

```bat
rem shim\include\test_utils\test_utils.h contains:  #pragma once  /  #include <gtest/gtest.h>
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set GT=<openvino>\thirdparty\gtest\gtest\googletest
cl /nologo /std:c++17 /EHsc /MD /O1 /c /I%GT%\include /I%GT% %GT%\src\gtest-all.cc %GT%\src\gtest_main.cc
cl /nologo /std:c++17 /EHsc /MD /O1 /W4 /permissive- /Ishim\include /I%GT%\include /I<branch>\src\plugins\intel_gpu\src\graph\impls ^
   <branch>\src\plugins\intel_gpu\src\graph\impls\ocl_v2\moe\moe_otd_pack.cpp ^
   <branch>\src\plugins\intel_gpu\tests\unit\test_cases\moe_offload_pack_publish_test.cpp ^
   gtest-all.obj gtest_main.obj /Fe:moe_otd_pack_tests.exe
moe_otd_pack_tests.exe
moe_otd_pack_tests.exe --gtest_repeat=50 --gtest_brief=1
```

The same recipe with `moe_offload_pack_test.cpp` runs the existing pack suite.

### In the real test binary (NOT RUN)

```powershell
cmake --build <build-dir> --target ov_gpu_unit_tests --parallel 8
& <ov_gpu_unit_tests.exe> '--gtest_filter=moe_otd_pack.*:moe_otd_pack_publish.*'
& <ov_gpu_unit_tests.exe> '--gtest_filter=moe_otd_pack_publish.*' --gtest_repeat=200
```

### Linux (`BLOCKED_ENV` for the author)

```bash
g++ -std=c++17 -O1 -Wall -Wextra -pthread -Ishim/include -I$GT/include -I$GT -I<branch>/src/plugins/intel_gpu/src/graph/impls \
  $GT/src/gtest-all.cc $GT/src/gtest_main.cc \
  <branch>/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_pack.cpp \
  <branch>/src/plugins/intel_gpu/tests/unit/test_cases/moe_offload_pack_publish_test.cpp -o publish_tests
./publish_tests && ./publish_tests --gtest_repeat=200
```

Expected on Linux: all tests take the "replaced" branch of their invariants, and the stress test has **zero** reader failures.

## 6. Windows 11 / Arc 140V local agent instructions

1. `git fetch origin feature/otd-pack-atomic-publish-cloud-20261022`, check out detached, `git status --short` clean, `git diff 479219245f6f10b670367ad35af16513d72d2091 --stat` shows only the four paths above.
2. Run the standalone harness first (no GPU, seconds), then the real `ov_gpu_unit_tests` filters above.
3. Manual check, no GPU inference needed: start any process that opens a pack through `ov::util` (or hold it open from PowerShell with `[System.IO.File]::Open(path,'Open','Read','ReadWrite')`), run a `GPU_MOE_OTD_PACK_MODE=build` compile that must rebuild that layer, and confirm a loud `cannot publish ... destination left unchanged` error and an unchanged `Get-FileHash`.
4. Do not reboot, change drivers, or overwrite global environment variables. Use a scratch pack directory, never the production pack set.

## 7. Linux / P720 parity checks

- Compile and run the Linux recipe above (GCC and Clang).
- Confirm with `stat -c %i` that a rebuild under a held reader replaces the inode while the reader keeps reading the old one.
- Case-sensitive paths: pack file names are lowercase hex, no change.
- No GPU equivalence is implied by any Linux result.

## 8. Performance acceptance

Serving path unchanged, so no serving benchmark is needed for this PR alone. Build mode: compare wall time of a full `build` pass before and after (the fsync cost), on the same disk, after the operator lifts the compute pause.

## 9. Independent review assignment

- **Reviewer 1 (C++/filesystem):** verify P1-P4 against MSVC STL `__std_fs_rename` and libstdc++ `rename`; check error-path ordering; check symlink and reparse-point behavior on both OSes; assess the TOCTOU note.
- **Reviewer 2 (build):** Windows MSVC and Linux GCC/Clang builds; check that `<windows.h>` with `WIN32_LEAN_AND_MEAN`/`NOMINMAX` causes no macro collisions in this TU; trial merge with CLD-01.
- **Reviewer 3 (tests):** run the suites in section 5 on both OSes and add Linux-specific checks (inode replacement, read-only directory).
- **Reviewer 4:** build-mode wall time only (section 8), after authorization.
- Order: CLD-01 and CLD-02 can be reviewed in parallel; if both are accepted, merge CLD-01 first, then rebase CLD-02.

**LOCAL-AGENT-BOOTSTRAP:** You are one reviewer above. Fetch only this branch and this file. Do not trust the author's results: rerun the red/green commands yourself (for the red run, check out the base `moe_otd_pack.cpp` with the new test file). Report commit SHA, exact command, exit code and any objection with file/line. Do not merge, do not touch C6 or production packs, and do not run GPU workloads during the operator pause.

## 10. STOP / rollback / open questions

- Rollback: revert this branch's commit (source-only). There are no format or runtime flags to change, and packs built by either version are byte-identical.
- STOP if any test shows a missing/truncated destination on POSIX, a deleted non-file destination, or a return with an unverified destination.
- Open questions for the operator: (a) should `PackLayout::open` retry the Windows transient `cannot open` (bounded, configure-time only)? (b) should a follow-up PR on top of CLD-01 pin reader identity (section 4)? (c) should stale `*.tmp-*` cleanup be added under an explicit builder lock?
