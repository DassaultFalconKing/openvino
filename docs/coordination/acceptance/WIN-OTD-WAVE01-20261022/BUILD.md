# BUILD

## Toolchain (Windows 11)
- CMake 4.3.3 / Ninja 1.13.2 / MSVC 14.44.35207 (VsDevCmd `-arch=x64`)
- `cmake -G Ninja -DENABLE_TESTS=ON -DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=OCL -DCMAKE_BUILD_TYPE=Release` (CPU/NPU/Python/samples/JS and most frontends OFF)
- `cmake --build <build> --target ov_gpu_unit_tests --parallel 8` → exit **0**

## Binary bind
- Artifact: `ov_gpu_unit_tests.exe` (Release)
- Size: 87942144 bytes
- SHA256: `92255A40A250B2C00076529F7812558FFE0D090B97384E10052D5560EA515B9B`
- UTC mtime: 2026-10-22T17:39:51Z
- Source HEAD at build: `99ee2c2c96044317469f8e518354c4aa6313e98c`
- Real link (not `/Zs` syntax-only).
