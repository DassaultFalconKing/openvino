# Словарь сокращений — GEMMAMONSTER / OTD / UMA (Windows)

Краткий глоссарий терминов из Wave-01, UMA path B, P0-B и model A/B. Не нормативный RFC — рабочий cheat-sheet.

---

## Продукт / репозитории

| Термин | Расшифровка |
|--------|-------------|
| **OTD** | On-demand / offload expert weights: MoE-веса частично на диске, resident-слоты на GPU/USM, pack v1 + LRU |
| **OVMS** | OpenVINO Model Server (`gemmamonster_model_server_OVMS`) |
| **C6** | Golden / production cohort OVMS+runtime (immutable в research-волнах) |
| **GenAI** | OpenVINO GenAI runtime (LLM/CB pipelines) |
| **IR** | Intermediate Representation модели OpenVINO |
| **MoE** | Mixture of Experts |
| **MTP** | Multi-Token Prediction / speculative draft path |
| **VLM_CB** | Vision-Language + Continuous Batching pipeline type в OVMS |
| **CLD-0x** | Cloud/cohort work packages (CLD-01 pack, CLD-02…, CLD-04 scheduler — позже) |

---

## Железо / память / драйвер

| Термин | Расшифровка |
|--------|-------------|
| **UMA** | Unified Memory Architecture — CPU и iGPU делят физическую DRAM |
| **CacheCoherentUMA** | D3D12-флаг: на этом хосте `false` → не полагаться на когерентность без явных fence |
| **USM** | Unified Shared Memory (OpenCL `cl_intel_unified_shared_memory`) |
| **usm_device** | USM-аллокация на device (типичный path A resident) |
| **usm_host** | USM host-visible аллокация (scratch path B resident) |
| **WDDM** | Windows Display Driver Model — бюджеты shared/local GPU memory |
| **NEO** | Intel OpenCL GPU runtime (NEO) |
| **Arc 140V** | Целевой iGPU на этой машине (MSI Claw / Core Ultra) |
| **LPDDR** | Общая системная память CPU+GPU (не «лишняя VRAM») |

---

## Path A / Path B / pack

| Термин | Расшифровка |
|--------|-------------|
| **Path A** | Pack → staging → verify → `copy_from` в `usm_device` → publish (`set_filled`) |
| **Path B** | Pack → staging → verify → CPU scatter (`memcpy`) в `usm_host` → fence → publish; **не** zero-copy с NVMe |
| **OTD_RESIDENT_USM_HOST** | Env-флаг scratch: `1` включает path B alloc; **default off** |
| **OFFLOAD_RATIO** | % expert-весов «на диске»; resident slots ≈ `N×(100−R)/100` (memory_safe часто **20**) |
| **pack v1** | Формат packed MoE expert records (девять тензоров / expert) |
| **require / build** | PackMode: собрать pack vs требовать готовый каталог |
| **LRU** | Least-Recently-Used кэш resident expert → slot |
| **set_filled** | Publish слота в LRU после успешного fill (после fence на path B) |
| **pre-write ownership** | Drain/finish **до** CPU overwrite слота (P0-B) |
| **post-write fence** | `flush`+`finish` **после** scatter, чтобы GPU увидел новые байты |
| **staging** | CPU-буфер записи с диска + checksum verify до scatter/copy |

---

## Вычисления / бенчи

| Термин | Расшифровка |
|--------|-------------|
| **GEMM** | GEneral Matrix Multiply (плотная матрица×матрица) |
| **GEMV** | Matrix×Vector (часто decode / малый M) |
| **M** | Leading dim батча/токенов в GEMM/GEMV (у нас часто `1,2,4,8,11`) |
| **TTFT** | Time To First Token |
| **TPOT** | Time Per Output Token |
| **p50 / p95 / p99** | Перцентили латентности |
| **A/B** | Сравнение path A vs path B (или baseline vs candidate) при том же SHA/драйвере |
| **C1 / C2 / C4** | Concurrent sequences в OVMS/CB: 1 / 2 / 4 одновременных запроса |
| **cold miss** | Expert не в LRU → disk/pack fill |
| **hit** | Expert уже в resident slot |
| **digest kernel** | Малый GPU-kernel «прочитать байты слота» (fixture, не GEMM) |

---

## Гейты / статусы / процесс

| Термин | Расшифровка |
|--------|-------------|
| **COMPUTE_GO** | Явное разрешение оператора на **живой GPU** прогон (не env OpenVINO). Scope всегда узкий |
| **HOST** | Unit/host тесты без model inference / без soak |
| **GPU_LIVE** | Короткие kernel/gtest на Arc под COMPUTE_GO |
| **HOST_PASS** | Host-матрица зелёная |
| **GPU_KERNEL_PASS** | Короткие GPU-kernel тесты зелёные (≠ full model) |
| **MODEL_PASS** | Реальный model serve/inference acceptance |
| **PERF_PASS** | Perf-гейт (representative GEMM / tok/s) пройден |
| **NOT_RUN** | Не запускалось (нет GO / не дошли) |
| **BLOCKED** | Заблокировано (нет GO, FAIL, нет машины…) |
| **LINUX_DEFERRED_NO_MACHINE** | Linux-порт отложен — нет Linux-стенда |
| **P0 / P1 / P2** | Приоритет гейтов roadmap (P0-B = in-flight slot safety) |
| **Wave-01** | Windows host acceptance волна (PR #7 linked tests) |
| **UMA-00/01/02/03** | Исследование UMA → design → host harness → bandwidth fixtures |
| **P0-B** | Correctness: CPU не пишет в slot, пока GPU его читает + visibility |
| **B_SAFE_FOR_MODEL_AB** | Path B корректен для **авторизованного** model A/B |
| **B_BLOCKED** | Race/visibility не закрыты или тест FAIL |
| **B_PERF_REJECTED** | Корректен, но GEMM/perf не оправдывает path B |
| **SWEETSPOT** | OFFLOAD_RATIO ∈ {20, 25, 30} pack-suitability fixtures |
| **Draft PR** | Не merge без operator; #7 Wave-01, #8 path B scratch, OVMS #23 roadmap |

---

## Тестовые ID (часто встречаются)

| ID | Смысл |
|----|--------|
| **U2-P / N / L / A** | UMA-02 host: Parity / Negatives / LRU / Allocator |
| **U2-G01–G03** | UMA-02 GPU_LIVE short (kernel read, cold miss, eviction lease) |
| **SS01–SS04** | Sweet-spot ratio formula / pack hits / overflow / path A↔B parity@20 |
| **P0B-O1 / O2** | Overlap same-queue / two-queue |
| **P0B-V1** | Visibility новых USM host writes |
| **P0B-L1** | No false LRU hit + reuse parity |

---

## Быстрые якоря путей

| Что | Где |
|-----|-----|
| OpenVINO scratch | `C:\gm\otd-research\wt-integration` (`scratch/uma-02-host-harness-20261022`) |
| Candidate runtime (не трогать) | `C:\gm\otd-research\runtime\otd` |
| Research model | `C:\llm\models\ornith-35b-ov` |
| Orchestration evidence | `C:\gm\orchestration\` |
| Этот словарь | `docs/coordination/GLOSSARY.md` |
