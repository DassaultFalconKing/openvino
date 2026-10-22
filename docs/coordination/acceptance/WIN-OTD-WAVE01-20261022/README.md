# WIN-OTD-WAVE01-20261022 — Windows host acceptance

**Integration:** `integration/otd-wave01-win-acceptance-20261022`  
**Baseline:** `479219245f6f10b670367ad35af16513d72d2091` (`research/ov-gpu-otd-cohort-b5bfb148`)  
**Linked `ov_gpu_unit_tests` SHA256:** `92255A40A250B2C00076529F7812558FFE0D090B97384E10052D5560EA515B9B`  
**Host combined gtest filter:** **47/47 PASS** (pack+publish+trace+provider)  
**Python `test_otd_sim`:** **33/33 PASS**  
**GPU live:** GPU_PENDING_AUTHORIZATION / NOT_RUN (no COMPUTE_GO)  
**Linux:** LINUX_DEFERRED_NO_MACHINE  
**Recommendation:** WINDOWS_ACCEPTED_WITH_RESTRICTIONS

| Doc | Purpose |
|-----|---------|
| SOURCE-IDENTITY.md | Pins, PR HEADs, merge order |
| BUILD.md | Toolchain + binary bind |
| TEST-MATRIX.md | Gate table |
| FINDINGS.md | Audits + residual risks |
| RACE-TESTS.md | Publish repeat=200 |
| GPU-ACCEPTANCE.md | PENDING_AUTHORIZATION |
| PERFORMANCE.md | NOT_RUN |
| HANDOFF.md | OVMS next action |
