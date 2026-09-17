# -*- coding: utf-8 -*-
"""
Run every test suite and print one summary line per suite.

    pipeline integrity   tests.test_pipeline            (original 75 checks)
    alignment            tests.test_alignment, all 7 words
    statistics           tests.test_stats
    table 4              tests.test_probe_geometry
    table 5              tests.test_intervention_tables

Exit code 1 if any suite fails.

Run:  python -m tests.run_all
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]

SUITES = [("pipeline integrity", ["tests.test_pipeline"])]
SUITES += [(f"alignment ({w})", ["tests.test_alignment", "--word", w]) for w in WORDS]
SUITES += [("statistics", ["tests.test_stats"]),
           ("table 4 probe-geometry", ["tests.test_probe_geometry"]),
           ("table 5 intervention", ["tests.test_intervention_tables"])]


def main():
    failed = []
    total_pass = total_fail = 0
    print(f"{'suite':<28}{'result':>28}")
    print("-" * 56)
    for name, args in SUITES:
        r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)
        m = re.findall(r"(\d+) passed, (\d+) failed", r.stdout)
        if m:
            p, f = map(int, m[-1])
            total_pass, total_fail = total_pass + p, total_fail + f
            summary = f"{p} passed, {f} failed"
        else:
            summary = "no summary line"
        ok = r.returncode == 0 and bool(m) and int(m[-1][1]) == 0
        print(f"{name:<28}{summary:>24}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)
            tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
            print("    " + "\n    ".join(tail))
    print("-" * 56)
    print(f"{'total':<28}{f'{total_pass} passed, {total_fail} failed':>24}")
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
