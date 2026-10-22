# UMA-02 GPU_LIVE (short) — 2026-10-22

See also local mirror: `C:\gm\orchestration\uma-02-windows-20261022\evidence\gpu-live\`

| ID | Class | Result |
|---|---|---|
| U2-G01 | `GPU_KERNEL_PASS` | **PASS** (kernel digests path A == path B on Arc 140V) |
| U2-G02 | `GPU_KERNEL_PASS` | **PASS** (cold-miss counters; no win claim) |
| U2-G03 | `GPU_KERNEL_PASS` | **PASS** — see `../UMA-03-CYCLE-20261022/G03.md` |

Pinned at close of sweetspot work:

| Item | Value |
|---|---|
| Code SHA | `d82f5f573dfb2b30a4cbf35b1cb8f75e4066c621` |
| `ov_gpu_unit_tests.exe` SHA256 | `ED14876C4C36A7316F0ABA25B609CD72CE82EE728A958A427FCF3C76EE729ACE` |

OpenCode (`opencode/glm-5.3-flash-free`, session `ses_e9b1b3eb7ffe5eklz6zhRlAG8l`) hit step limit after diagnosing G01 test bug (`build_kernels(&src)`); coordinator reverted TEMP-DIAG in `ocl_kernel_builder.hpp`, rebuilt, re-ran — exit 0.

**Follow-on:** P0-B in-flight ownership/visibility — `../P0B-INFLIGHT-20261022/` (required before model A/B).

**Not** production enablement. `OTD_RESIDENT_USM_HOST` default off. No merge.
