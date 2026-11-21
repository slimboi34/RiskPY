"""Throughput of the compiled core: Rust (0.9.0) against C++ (0.8.0), same machine, same session.

Each workload is timed in a fresh interpreter for each core, the two cores alternating,
and the median of the rounds is reported. Times include building the Python list the
methods return, as a user sees them.

    python throughput.py --cpp /venv-cpp/bin/python --rust /venv-rust/bin/python [--rounds 7]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys

WORKLOADS = [
    ("aggregate loss λ=5, lognormal", "simulate_aggregate_loss", (5.0, 8.0, 0.5), 200_000),
    ("aggregate loss λ=140 (PTRS)", "simulate_aggregate_loss", (140.0, 9.3, 1.1), 50_000),
    ("life portfolio n=1000, vol 0.1", "simulate_life_portfolio", (1000, 0.01, 0.1, 100_000.0), 200_000),
    ("life portfolio n=50000 (BTPE)", "simulate_life_portfolio", (50_000, 0.004, 0.25, 100_000.0), 200_000),
    ("economic path, 12 steps", "simulate_economic_path", (100.0, 0.05, 0.2, 12), 200_000),
    ("health claims λ=10, d=0.3", "simulate_health_claims", (10.0, 0.3, 2.0, 500.0), 200_000),
    ("catastrophe p=0.05", "simulate_catastrophe_loss", (0.05, 1e6, 1.8), 1_000_000),
]

SNIPPET = """
import json, sys, time
import riskpy.cpp_underwriter as core
method, params, trials = json.loads(sys.argv[1])
sim = core.MonteCarloSimulator(trials, 12345)
f = getattr(sim, method)
f(*params)                                   # warm-up
best = []
for _ in range(3):
    t0 = time.perf_counter(); f(*params); best.append(time.perf_counter() - t0)
print(json.dumps(min(best)))
"""

FFT_SNIPPET = """
import json, sys, time
import riskpy.cpp_underwriter as core
sev = [0.0] + [1.0 / 200] * 200
core.FourierTransform.compound_poisson_pmf(sev, 50.0)
best = []
for _ in range(3):
    t0 = time.perf_counter(); core.FourierTransform.compound_poisson_pmf(sev, 50.0); best.append(time.perf_counter() - t0)
print(json.dumps(min(best)))
"""


def time_once(python, workload):
    if workload is None:
        out = subprocess.run([python, "-I", "-c", FFT_SNIPPET], capture_output=True, text=True, check=True).stdout
    else:
        _, method, params, trials = workload
        out = subprocess.run([python, "-I", "-c", SNIPPET, json.dumps([method, params, trials])],
                             capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--rust", required=True)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--json")
    ns = ap.parse_args()
    rows = []
    for w in WORKLOADS + [None]:
        label = w[0] if w else "compound_poisson_pmf, λ=50, 201-point severity"
        trials = w[3] if w else None
        cpp, rust = [], []
        for _ in range(ns.rounds):
            cpp.append(time_once(ns.cpp, w))
            rust.append(time_once(ns.rust, w))
        c, r = statistics.median(cpp), statistics.median(rust)
        rows.append({"workload": label, "trials": trials, "cpp_ms": c * 1e3, "rust_ms": r * 1e3, "speedup": c / r})
        print(f"{label:<48} {trials or '':>9}  C++ {c * 1e3:9.2f} ms   Rust {r * 1e3:9.2f} ms   "
              f"C++/Rust {c / r:5.2f}x", flush=True)
    if ns.json:
        with open(ns.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
