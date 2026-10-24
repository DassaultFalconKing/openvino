"""Fault-inject native drain failures in a child, never the test runner."""
import subprocess
import sys

for mode in ("--failed-drain", "--failed-ocl-drain", "--both-failed", "--good-drain"):
    child = subprocess.run([sys.argv[1], mode], capture_output=True, text=True, timeout=30)
    expected = 0 if mode == "--good-drain" else 86
    if child.returncode != expected:
        raise RuntimeError((mode, child.returncode, child.stdout, child.stderr))
    for marker in ("ONEDNN_DRAIN_ATTEMPTED", "OCL_DRAIN_ATTEMPTED"):
        if marker not in child.stderr:
            raise RuntimeError((mode, "missing drain", child.stderr))
    if expected and "OTD_FATAL_UNVERIFIED_GPU_COMPLETION" not in child.stderr:
        raise RuntimeError((mode, "missing fail-stop diagnostic"))
print("4 native drain fault-injection cases PASS")
