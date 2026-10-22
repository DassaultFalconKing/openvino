# TEST-MATRIX

Bound to source `99ee2c2c96044317469f8e518354c4aa6313e98c` and binary SHA256 `92255A40A250B2C00076529F7812558FFE0D090B97384E10052D5560EA515B9B`.

| Gate | Result | Detail |
|------|--------|--------|
| BUILD | PASS | linked `ov_gpu_unit_tests` exit 0 |
| PACK | PASS | `moe_otd_pack.*` 16/16 |
| PUBLISH | PASS | 5/5; `--gtest_repeat=200` exit 0 |
| TRACE | PASS | C++ 13/13; Python `test_otd_sim` 33/33 |
| HANDLE | PASS | `moe_expert_weight_provider.*` 13/13 |
| Combined filter | PASS | 47/47 exit 0 |
| GPU live | NOT_RUN | GPU_PENDING_AUTHORIZATION |
| PERF | NOT_RUN | mandate |
| Linux | LINUX_DEFERRED_NO_MACHINE | |

Filter used: `moe_otd_pack.*:moe_otd_pack_publish.*:moe_otd_trace.*:moe_expert_weight_provider.*`
