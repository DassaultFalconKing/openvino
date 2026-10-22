# FINDINGS

## Audits (PASS with RISKs)
- C++: verify-before-copy + same-handle snapshot preserved; staging not MT-safe by design.
- FS: atomic rename + FlushFileBuffers + fail-closed replace; stale temps; Windows `cannot open` without retry.
- Trace: v1 compat + `policy_evidence_eligible`; incomplete-call miss gap residual.

## Fixes on integration branch
| Item | Commit | Notes |
|------|--------|-------|
| Sync pinned PR#1 docs tip | `99ee2c2c` | docs-only cherry-pick of `0ba38e44` |

No code-defect fix commits required for linked host PASS.

## Residual risks
1. Windows `cannot open` under held-reader replace (no retry).
2. Stale `*.tmp-*` after crash.
3. Unmeasured FNV checksum CPU cost on miss-heavy loads.
4. No COMPUTE_GO → no live Arc/OTD acceptance.
