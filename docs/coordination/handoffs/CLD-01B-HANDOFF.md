# CLD-01B-HANDOFF — Pin pack metadata and serving reader to one file handle

Status: `IMPLEMENTED_NOT_RUN`, awaiting independent native OpenVINO review, build, and GPU test. **Not accepted.** This is a narrow source-bound correctness follow-up to Draft [CLD-01/#1](https://github.com/DassaultFalconKing/openvino/pull/1), not CLD-04 scheduler work.

## Identity, dependencies, changed files

- Repo: `DassaultFalconKing/openvino`. Work branch: `fix/otd-reader-identity-pin-20261022`.
- Parent SHA: `f288a5b90f8011eb73c5aab3409108c05b313237` (CLD-01 Draft PR #1 HEAD as observed when this work began).
- First code commit: `359445cd40413b9bb36f16ca7300782fda595462`. New docs/HEAD SHA: inspect the branch after the doc commit.
- Original GPU cohort source: `research/ov-gpu-otd-cohort-b5bfb148` @ `479219245f6f10b670367ad35af16513d72d2091`.
- OVMS `integration/qwm01-otd-pack-20261022` @ `746dc833dc3bcccf64fabdb469feda9260582bae`, golden C6 `ba6ec792c3121850fea167fa7d7e190b25fcdc76`: unchanged.
- Files: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_pack.hpp/.cpp`, `expert_weight_providers.cpp`, `src/plugins/intel_gpu/tests/unit/test_cases/moe_offload_pack_test.cpp`, this handoff.

## What changed

Before, `PackLayout::open(path)` loaded metadata with an ephemeral `std::ifstream`, but `ParallelWeightReader` did not open its own OS handle until the first miss. Concurrent `rename` between the two meant checksum expectations might belong to an old pack while payload reads came from the replacement.

- New `PackLayout::matches_reader_snapshot(file_size, read)`: reads header, expert-to-position placement table and logical-expert checksum table through an already-open **serving handle**; compares all against validated layout.
- `OffloadExpertWeightProvider::configure_pack` now calls `ParallelWeightReader::file_size()` to force opening and owning its OS handle **before** accepting the pack. It checks the snapshot via that same reader's positional reads and stores that same reader in `_pack_reader`.
- Snapshot mismatch, handle-read failure or short file: `require` throws before GPU activation; `use` preserves its existing explicit unpacked fallback (during configuration only). No partial `_pack` activation.
- Test `moe_otd_pack.serving_reader_snapshot_is_bound_to_validated_metadata`: valid layout; exactly 3 metadata reads (zero expert payload reads); short file; altered header, placement, checksum and injected read exception.
- No pack v1 layout changes, no new disk reads per **cache miss**, no new cache tier, no router, KV, speculative or model work.

## Key invariants and remaining risks

1. Header, placement and expected checksums correspond to the same OS handle used by later expert reads. On POSIX an atomic rename leaves an open reader on its original inode; on Windows an open handle may prevent overwrite. In-place writes can still happen and are subject to the per-miss checksum check.
2. The initial short-lived `PackLayout::open(path)` remains, so the snapshot comparison costs **three metadata reads at configure-time**, O(experts) extra read bytes (12*N + header) and two temporary vectors; it is not free. Runtime still uses one positional read per miss and O(payload) FNV verification from CLD-01.
3. No cryptographic identity/authentication: a malicious writer able to forge matching hashes is outside the FNV protection.
4. This does not fix the CLD-02 **Windows transient** `cannot open` during a concurrent replacement, nor the CLD-02 temp-file exclusive-create TOCTOU. No retries were introduced. The product-level decision remains to prebuild packs and avoid concurrent build+serving until reviewed.
5. Failure of an expert read or GPU copy after configuration is not automatically recoverable; do not promote to production without real runtime and cache-failure tests.
6. The snapshot method uses a `std::function` only at configuration, not in hot inference.
7. `std::filesystem::file_size` via reader pins a handle, but provider concurrency still requires independent lifetime/thread-safety review. Do not assume a single `_pack_staging` can safely serve concurrent fills.

## Native local verification, not performed in cloud

### Windows/Arc 140V (PowerShell, source/build/test stage first; no inference unless authorized)

```powershell
git fetch origin fix/otd-reader-identity-pin-20261022
git switch --detach origin/fix/otd-reader-identity-pin-20261022
git status --short
git rev-parse HEAD
cmake -S . -B build-cld01b -G Ninja -DENABLE_TESTS=ON -DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=OCL
cmake --build build-cld01b --target ov_gpu_unit_tests --parallel 8
$exe = Get-ChildItem build-cld01b -Recurse -Filter ov_gpu_unit_tests.exe | Select-Object -First 1 -ExpandProperty FullName
& $exe '--gtest_filter=moe_otd_pack.*:moe_expert_weight_provider.*'
if ($LASTEXITCODE -ne 0) { throw "CLD-01B tests failed" }
```

Existing local build flags/VS developer shell take precedence over the illustrative CMake recipe. Confirm the loaded Intel GPU DLL is built from this exact HEAD. Run new snapshot test and CLD-01 corruption cases, then the four provider real-device tests. Validate: `require` mismatch fails before any GPU copy; `use` startup fallback only; no stale slot used after exception. Windows test should hold serving handle while trying concurrent pack replacement with a **scratch** directory.

### Linux / P720 parity (no GPU equivalence)

```bash
git fetch origin fix/otd-reader-identity-pin-20261022
git switch --detach origin/fix/otd-reader-identity-pin-20261022
cmake -S . -B build-cld01b -G Ninja -DENABLE_TESTS=ON -DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=OCL
cmake --build build-cld01b --target ov_gpu_unit_tests -j8
find build-cld01b -type f -name ov_gpu_unit_tests -perm -111
```

Verify read-after-rename inode pinning, GCC/Clang compilation, no partial metadata accepted, absent/truncated files, and controlled cross-process replacement. The published author has NOT executed these native tests.

## Reviewer assignments / bootstrap

- **C++ correctness:** independently audit `matches_reader_snapshot`, vector allocation bounds, handle lifetime, `PackIntegrityError`, and `PackMode` behavior. Check for concurrent provider reconfiguration and per-request sharing.
- **Build/ABI:** reproduce Windows MSVC + Linux GCC/Clang full targets, review `std::function` ABI scope and use of `ParallelWeightReader::file_size()`.
- **Negative unit:** run all pack and provider tests, add a cross-process rename/replace fixture. On Windows test transient cannot-open without false fallback; do not yet add unmeasured retry loops.
- **Performance:** **after operator permission**, time configure-time snapshot reads and per-miss checksum vs native r40/r75; record logical bytes vs physical OS NVMe bytes, TTFT, p95/99 TPOT and UMA pressure.

**LOCAL-AGENT-BOOTSTRAP:** Check out the exact HEAD of this PR independently, read this handoff and the four-file diff against parent `f288a5b`, then run only your assigned review. Include compiler/toolchain, exact command/exit code, file line and negative fixture with every finding. Do not inherit author's `PASS` claims, auto-merge, run full GPU soak, or change C6.

## Integration order and rollback

CLD-01/#1 → CLD-01B (this dependent PR) → CLD-02/#2 after rebase/test → CLD-03/#3 (independent). Before any integration do a combined native build + GPU-focused negative test run and fresh PR review. For rollback, revert only this follow-up commit or build the pinned CLD-01 parent. To compare unpacked weights, set `GPU_MOE_OTD_PACK_MODE=off` in a *test* run only. Never silently downgrade `require` in production. Stop on missing metadata, unexpected fallback, invalid slot contents, or shifted model outputs.

Status codes: `IMPLEMENTED_NOT_RUN` for new C++ changes/tests, `BUILD NOT_RUN`, `UNIT NOT_RUN`, `ARC_NOT_RUN`, `ACCEPTED=NO`.
