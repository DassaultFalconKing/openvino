# Cloud task: MoE decode 4× regression (2026.5 GPU plugin vs 2026.4)

## Source identity (authority)

- Repository: `https://github.com/DassaultFalconKing/openvino.git`
- Mutable ref (context): `codex/c6-head-20261019-openvino`
- Exact source SHA: `e5606d01c464dc125ab79183ca43c2042d0e2f67`
- Base ref: `codex/c6-engine-20261018`
- Stock compare: OpenVINO `2026.4.0-22930-61afcb26271` (`C:\llm\ovms` on the capture machine)
- Head runtime identity: `2026.5.0-5-591ab0b517a` plus later V1–V4 + cache-serialization commits on this branch

This orphan branch is **not** repository authority. Product source lives on `codex/c6-head-20261019-openvino`. Do not treat this root commit as a source branch.

## Scope

Pinpoint the micro-causal root of a **~4× MoE decode regression** on Lunar Lake Arc 140V:

| arm | OFFLOAD_RATIO | decode tok/s |
|---|---|---|
| stock 2026.4 | 20 | 31.8 / 32.2 / 33.0 |
| head plugin | 20 | 8.12 / 8.28 / 8.28 |
| head plugin | 0 | 8.39 / 8.36 / **8.99** (path-log run) |
| head, no parsers | 0 | 8.48 |

Model: Qwen3.6-35B-A3B-int4 (40 MoE layers, hidden=2048, 256 experts, topk=8, hybrid GDN + full attention). Harness: OVMS + GenAI continuous batching (`ab-run.ps1`).

## Already falsified (do not re-litigate without new evidence)

1. OTD hooks always-on — `OFFLOAD_RATIO=0` still ~8.4 tok/s (`ResidentExpertWeightProvider`).
2. Partial-upload `usm_device` — dead at ratio=0.
3. Build flags — Release `/O2`, `ENABLE_ONEDNN_FOR_GPU=ON`, `GPU_RT_TYPE=OCL`.
4. OVMS tool/reasoning parsers.
5. GEMV `.cl` kernel text — same as stock except a comment.
6. `supports_replay()=false` / record-replay.
7. Cross-plugin DLL swap (2026.4 plugin into 2026.5 core) — load error 127.
8. **`get_seq_len` / wrong GEMM path on decode** — `MOE_PATH_LOG` on SHA `e5606d01c464dc125ab79183ca43c2042d0e2f67` shows:

   - prefill: `token_num=55 thr=32 path=grouped shape=55,1,2048`
   - decode: `token_num=1 thr=32 path=gemv shape=1,1,2048`

   Status: **FAIL** for the “padded token_num > 32 → prefill grouped/onednn” hypothesis. Do **not** patch `get_seq_len`.

## Still open

1. GEMV *callers* around `exec_batched_gemv` (weight bind / expert gather / extra events) even though `.cl` text matches stock.
2. Non-MoE on the same step (paged-attention GENERATE / GDN / extra 1.22 GB compiled blob on head).
3. Host sync in `primitive_inst` / `ocl_stream` / OneDNN interop. GPU util sampler returned `n=0`.

## Expected outputs

- A ranked remaining hypothesis with a single next measurement (not a guess-patch).
- If a one-line plugin fix is proposed, it must be justified by new evidence, not by the falsified `token_num` theory.
- Keep statuses truthful: `FAIL`, `NOT RUN`, `UNKNOWN`, `LIVE_RUNTIME_REQUIRED`.

## Tests to run

- Replay interpretation of `evidence/runs/head-moe-path-20261023-004014/stderr.log` (`[MOE_PATH]` lines) plus the A/B `bench.json` files. **NOT RUN** on this orphan; logs are already captured.
- Any new GPU decode bench: **LIVE_RUNTIME_REQUIRED** (Lunar Lake iGPU + OVMS + the int4 model). The capture machine runtime and model weights are **not** in this branch.

## Known failures / limitations

- `head+stockgpu` arm failed (plugin ABI, error 127). No `bench.json`.
- GPU/CPU util sampler: `n=0` (no counters).
- Kernel cache (`*.cl_cache`, `*.blob`) and staged `openvino_intel_gpu_plugin.dll` / `ovms.exe` are **not** in this orphan (binaries / generated bulk).
- Product `.cpp` copies and the 2026.4→head plugin diff are omitted; read them from source SHA `e5606d01c464dc125ab79183ca43c2042d0e2f67` on `codex/c6-head-20261019-openvino`.
- Probe is env-gated (`MOE_PATH_LOG`). Harmless when unset.
