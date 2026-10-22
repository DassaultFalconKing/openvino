# UMA-02 GPU_LIVE (short) — 2026-10-22

See also local mirror: `C:\gm\orchestration\uma-02-windows-20261022\evidence\gpu-live\`

| ID | Result |
|---|---|
| U2-G01 | **PASS** (kernel digests path A == path B on Arc 140V) |
| U2-G02 | **PASS** (cold-miss counters; no win claim) |
| U2-G03 | **PASS** — see `../UMA-03-CYCLE-20261022/G03.md` |

OpenCode (`opencode/glm-5.3-flash-free`, session `ses_e9b1b3eb7ffe5eklz6zhRlAG8l`) hit step limit after diagnosing G01 test bug (`build_kernels(&src)`); coordinator reverted TEMP-DIAG in `ocl_kernel_builder.hpp`, rebuilt, re-ran — exit 0.

**Not** production enablement. `OTD_RESIDENT_USM_HOST` default off. No merge.
