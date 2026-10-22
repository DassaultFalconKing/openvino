# P0-B pins — 2026-10-22

| Item | Value |
|---|---|
| Branch | `scratch/uma-02-host-harness-20261022` |
| Draft PR | https://github.com/DassaultFalconKing/openvino/pull/8 |
| Pre-fix code SHA | `d82f5f573dfb2b30a4cbf35b1cb8f75e4066c621` |
| Pre-fix `ov_gpu_unit_tests.exe` SHA256 | `ED14876C4C36A7316F0ABA25B609CD72CE82EE728A958A427FCF3C76EE729ACE` |
| Post-fix code SHA | `13d56464c2af9e40b5bd27cbabe99d58784bfc6a` |
| Fix commit | `ed246cdf0252185fae6f16c6391afa63ec36404c` |
| Post-fix binary SHA256 | `E35E432C3879E903534FE74D41B83155AB04AD7AA974957D4B919024E2C1A8F7` |

## Prior gate status (reconciled)

| Gate | Class | Result |
|---|---|---|
| UMA-02 host P/N/L/A | `HOST_PASS` | PASS (prior) |
| U2-G01 / G02 / G03 | `GPU_KERNEL_PASS` | PASS — see `../UMA-02-HOST-20261022/GPU-LIVE.md` + UMA-03 |
| UMA-03 cycle | `GPU_KERNEL_PASS` | PASS — measured; no production win claim |
| Sweet-spot 20/25/30 | `HOST_PASS` | `SWEETSPOT_PASS` |
| P0B-O1 / O2 / V1 / L1 | `GPU_KERNEL_PASS` | *(this run)* |
| Model / OVMS A/B | `MODEL_PASS` | `NOT_RUN` until Phase 3 PASS + operator OK |

**Stale:** `UMA-02-HOST/VERDICT.md` previously said G01–G03 `BLOCKED_COMPUTE_GO` / PR body “GPU_LIVE BLOCKED”. Those are superseded by GPU-LIVE + UMA-03 + sweetspot PASS docs; this P0-B packet is the ownership/visibility gate before model A/B.
