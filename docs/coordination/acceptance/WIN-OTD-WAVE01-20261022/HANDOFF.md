# HANDOFF — Windows Wave01 → OVMS integrator

## Identity
- Branch: `integration/otd-wave01-win-acceptance-20261022`
- Baseline: `479219245f6f10b670367ad35af16513d72d2091`
- Source PRs: #1 #4 #2 #3 #5 (tips match pins)
- Binary SHA256: `92255A40A250B2C00076529F7812558FFE0D090B97384E10052D5560EA515B9B`
- GPU: PENDING_AUTHORIZATION | Linux: LINUX_DEFERRED_NO_MACHINE

## Host gates
BUILD/PACK/HANDLE/PUBLISH/TRACE/INTEGRATION (host): PASS.
GPU live / PERF: NOT_RUN.

## Recommendation
WINDOWS_ACCEPTED_WITH_RESTRICTIONS (host-only; GPU gates open).

## Single next action for OVMS integrator
Pin review to this Draft PR HEAD; do not enable production OVMS OTD pack paths or claim GPU acceptance until COMPUTE_GO and Arc live gates PASS on this SHA.

## Final evidence commit
- Evidence/docs commit: `f82fbaaad1efafbcadd9cc5f513bd3a8409b3cfd`

