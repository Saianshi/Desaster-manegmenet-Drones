"""
Validates that parallelising evaluation/compare.py's seed loop via
multiprocessing does not change results: run the same seed serially
(max_workers=1) and in parallel (max_workers>1, alongside other seeds so
there's real concurrency to race against), then diff the metrics dicts.

Run standalone with:  python -m evaluation.test_determinism
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from evaluation.compare import run_condition


def main():
    seed = params.EVAL_BASE_SEED
    other_seeds = [params.EVAL_BASE_SEED + i for i in range(1, 5)]  # padding for real parallelism
    n_resources = params.N_RESOURCES_STUB

    print("Running serially (max_workers=1)...")
    serial = run_condition("determinism/serial", "full_engine", [seed], n_resources, max_workers=1)[0]

    print("Running in parallel (max_workers=4, alongside 4 other seeds)...")
    parallel_batch = run_condition("determinism/parallel", "full_engine", [seed] + other_seeds, n_resources, max_workers=4)
    parallel = parallel_batch[0]  # seed is first in the input list; run_condition preserves input order

    print("\n=== Serial result ===")
    for k, v in serial.items():
        print(f"  {k}: {v}")
    print("\n=== Parallel result (same seed) ===")
    for k, v in parallel.items():
        print(f"  {k}: {v}")

    mismatches = {k: (serial[k], parallel[k]) for k in serial if serial[k] != parallel[k]}
    if mismatches:
        print(f"\nFAIL: {len(mismatches)} field(s) differ between serial and parallel runs of seed {seed}:")
        for k, (s, p) in mismatches.items():
            print(f"  {k}: serial={s}  parallel={p}")
        sys.exit(1)
    else:
        print(f"\nPASS: seed {seed} produced IDENTICAL metrics serially and in parallel.")


if __name__ == "__main__":
    main()
