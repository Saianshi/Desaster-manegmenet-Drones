"""
Phase 7 evaluation: full engine vs naive baseline vs informed operator,
feature ablations, and a resource-count sensitivity sweep.

Fairness matters throughout: every condition in a given comparison uses
the SAME set of seeds, so seed 1007's full-engine run and seed 1007's
baseline run see the IDENTICAL world, buildings, victims, and terrain --
only the decision-maker differs. simulation.harness.run_scenario is what
guarantees that (see its docstring).

This is a long run (~350 simulated scenarios at full scale). Three
robustness/performance measures earn their keep at that length:
  - MAX_RETRIES in run_condition absorbs a transient allocation error
    observed sporadically on this machine (see engine/belief.py's Phase 4
    notes) that has nothing to do with actual memory pressure.
  - A checkpoint file (output/eval_checkpoint*.json) is saved after every
    completed condition, and reloaded on startup -- a killed/interrupted
    run resumes instead of restarting from zero. This exists because an
    external session teardown killed a run partway through once already.
  - Seeds within a condition run in PARALLEL via multiprocessing (each
    seed is a fully independent simulation -- its own random.Random(seed)
    and its own World, nothing shared across processes). Engine
    construction is done via _build_engine's string-keyed lookup rather
    than closures/lambdas, because lambdas aren't picklable and
    multiprocessing on Windows uses spawn, which requires pickling the
    worker's target and arguments. Determinism is per-seed and doesn't
    depend on worker count or completion order -- see
    evaluation/test_determinism.py.

Run standalone with:  python -m evaluation.compare [--quick] [--workers N]
--quick uses far fewer seeds/resource levels, for fast iteration while
developing this script -- NOT what the reported checkpoint numbers
should come from (those need the full EVAL_N_SEEDS=30, per spec).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from baseline.informed_operator import InformedOperatorEngine
from baseline.nearest_first import NearestFirstEngine
from engine.engine import Engine
from simulation.harness import run_scenario

ABLATION_CONDITION_KEYS = [
    "full_engine", "minus_cascade_bonus", "minus_silence_as_signal",
    "minus_rescuer_risk", "informed_operator", "pure_baseline",
]
SENSITIVITY_STRATEGY_KEYS = ["full_engine", "informed_operator", "pure_baseline"]

MAX_RETRIES = 5  # [DESIGN] a transient numpy _ArrayMemoryError (observed
                  # sporadically on this machine, unrelated to array size --
                  # see engine/belief.py's Phase 4 notes) shouldn't be able
                  # to kill a long unattended evaluation run


def build_engine(condition_key: str, seed: int):
    """String-keyed, module-level (picklable) engine constructor -- see
    this module's docstring for why lambdas don't work here."""
    if condition_key == "full_engine":
        return Engine(seed=seed)
    if condition_key == "minus_cascade_bonus":
        return Engine(seed=seed, use_cascade_bonus=False)
    if condition_key == "minus_silence_as_signal":
        return Engine(seed=seed, use_silence_as_signal=False)
    if condition_key == "minus_rescuer_risk":
        return Engine(seed=seed, use_rescuer_risk=False)
    if condition_key == "informed_operator":
        return InformedOperatorEngine(seed=seed)
    if condition_key == "pure_baseline":
        return NearestFirstEngine(seed=seed)
    raise ValueError(f"unknown condition_key: {condition_key}")


def run_one(condition_key: str, seed: int, n_resources: int) -> dict:
    """The actual unit of work a worker process executes. Must be a
    module-level function (not a closure) to be picklable under spawn."""
    engine = build_engine(condition_key, seed)
    return run_scenario(engine, seed=seed, n_resources=n_resources)


def _run_one_args(args):
    condition_key, seed, n_resources = args
    return run_one(condition_key, seed, n_resources)


def run_condition(name: str, condition_key: str, seeds: list, n_resources: int, max_workers: int) -> list:
    """
    Runs every seed for one condition in parallel (one process pool per
    condition, sized to max_workers). Retries only the seeds that hit a
    transient MemoryError, in a fresh round, up to MAX_RETRIES rounds.
    Returns results in the SAME order as `seeds`, regardless of which
    worker computed which or how long each took -- this is what keeps
    the run reproducible independent of process count/scheduling.
    """
    results_by_seed = {}
    to_run = list(seeds)
    t0 = time.time()

    for attempt in range(1, MAX_RETRIES + 1):
        if not to_run:
            break
        failed = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(_run_one_args, (condition_key, seed, n_resources)): seed
                for seed in to_run
            }
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    m = future.result()
                    results_by_seed[seed] = m
                    elapsed = time.time() - t0
                    print(f"  [{name}] seed {seed} ({len(results_by_seed)}/{len(seeds)}) -> "
                          f"rescued_people={m['rescued_people']}  ({elapsed:.0f}s elapsed)", flush=True)
                except MemoryError:
                    failed.append(seed)
        if failed:
            print(f"  [{name}] {len(failed)} seed(s) hit a transient error on attempt "
                  f"{attempt}/{MAX_RETRIES}, retrying: {failed}", flush=True)
            time.sleep(5)
        to_run = failed

    if to_run:
        print(f"  [{name}] {len(to_run)} seed(s) FAILED after {MAX_RETRIES} attempts, skipping: {to_run}", flush=True)

    return [results_by_seed[s] for s in seeds if s in results_by_seed]


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            cp = json.load(f)
        print(f"Loaded checkpoint from {path} (ablation: {list(cp.get('ablation', {}).keys())}, "
              f"sensitivity strategies: {list(cp.get('sensitivity', {}).keys())})", flush=True)
        return cp
    return {"ablation": {}, "sensitivity": {}}


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(checkpoint, f)
    tmp.replace(path)  # atomic-ish: never leaves a half-written checkpoint


def run_ablation(seeds, n_resources, checkpoint: dict, checkpoint_path: Path, max_workers: int) -> dict:
    ablation = checkpoint.setdefault("ablation", {})
    for name in ABLATION_CONDITION_KEYS:
        cached = ablation.get(name)
        if cached and cached.get("seeds") == seeds:
            print(f"=== Ablation condition: {name} -- RESUMED from checkpoint "
                  f"({cached['n_seeds_succeeded']}/{cached['n_seeds_requested']} seeds) ===", flush=True)
            continue

        print(f"=== Ablation condition: {name} ({len(seeds)} seeds, {max_workers} workers) ===", flush=True)
        runs = run_condition(name, name, seeds, n_resources, max_workers)
        rescued = [r["rescued_people"] for r in runs]
        alive = [r["n_alive_people"] for r in runs]
        mean_r, std_r = mean_std(rescued)
        mean_a, std_a = mean_std(alive)
        ablation[name] = {
            "rescued_people_mean": mean_r, "rescued_people_std": std_r,
            "alive_people_mean": mean_a, "alive_people_std": std_a,
            "raw_rescued_people": rescued,
            "n_seeds_requested": len(seeds),
            "n_seeds_succeeded": len(runs),
            "seeds": seeds,
        }
        print(f"  -> {name}: rescued_people = {mean_r:.2f} +/- {std_r:.2f}  ({len(runs)}/{len(seeds)} seeds succeeded)", flush=True)
        save_checkpoint(checkpoint, checkpoint_path)
    return ablation


def run_sensitivity(seeds, resource_counts, checkpoint: dict, checkpoint_path: Path, max_workers: int) -> dict:
    sensitivity = checkpoint.setdefault("sensitivity", {})
    for strat_name in SENSITIVITY_STRATEGY_KEYS:
        sensitivity.setdefault(strat_name, {})
        for n_resources in resource_counts:
            key = str(n_resources)
            cached = sensitivity[strat_name].get(key)
            if cached and cached.get("seeds") == seeds:
                print(f"=== Sensitivity: {strat_name}, n_resources={n_resources} -- RESUMED from checkpoint "
                      f"({cached['n_seeds_succeeded']}/{cached['n_seeds_requested']} seeds) ===", flush=True)
                continue

            print(f"=== Sensitivity: {strat_name}, n_resources={n_resources} ({len(seeds)} seeds, {max_workers} workers) ===", flush=True)
            runs = run_condition(f"{strat_name}/{n_resources}res", strat_name, seeds, n_resources, max_workers)
            rescued = [r["rescued_people"] for r in runs]
            mean_r, std_r = mean_std(rescued)
            sensitivity[strat_name][key] = {
                "rescued_people_mean": mean_r, "rescued_people_std": std_r,
                "raw_rescued_people": rescued,
                "n_seeds_requested": len(seeds), "n_seeds_succeeded": len(runs),
                "seeds": seeds,
            }
            print(f"  -> {strat_name} @ {n_resources} resources: rescued_people = {mean_r:.2f} +/- {std_r:.2f}  ({len(runs)}/{len(seeds)} seeds succeeded)", flush=True)
            save_checkpoint(checkpoint, checkpoint_path)
    return sensitivity


def plot_ablation(ablation: dict, n_seeds: int, out_path: Path) -> None:
    labels = ["Full engine", "- cascade\nbonus", "- silence-\nas-signal", "- rescuer\nrisk", "Informed\noperator", "Pure baseline\n(nearest-first)"]
    means = [ablation[k]["rescued_people_mean"] for k in ABLATION_CONDITION_KEYS]
    stds = [ablation[k]["rescued_people_std"] for k in ABLATION_CONDITION_KEYS]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#1E6FB5", "#4C9A4C", "#4C9A4C", "#4C9A4C", "#E8A33D", "#c0392b"]
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel(f"People rescued (mean +/- std, N={n_seeds} seeds)")
    ax.set_title("Ablation: contribution of each engine feature")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)


def plot_sensitivity(sensitivity: dict, resource_counts, n_seeds: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for strat_name, color, label in [
        ("full_engine", "#1E6FB5", "Full engine"),
        ("informed_operator", "#E8A33D", "Informed operator"),
        ("pure_baseline", "#c0392b", "Pure baseline (nearest-first)"),
    ]:
        means = [sensitivity[strat_name][str(n)]["rescued_people_mean"] for n in resource_counts]
        stds = [sensitivity[strat_name][str(n)]["rescued_people_std"] for n in resource_counts]
        ax.errorbar(resource_counts, means, yerr=stds, marker="o", capsize=5, color=color, label=label, linewidth=2)
    ax.set_xlabel("Number of resources (scarce -> plentiful)")
    ax.set_ylabel(f"People rescued (mean +/- std, N={n_seeds} seeds)")
    ax.set_title("Sensitivity: advantage vs resource scarcity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)


def _parse_workers(argv) -> int:
    import os
    default = max(1, (os.cpu_count() or 4) - 1)  # leave one core free
    if "--workers" in argv:
        idx = argv.index("--workers")
        return int(argv[idx + 1])
    return default


def main():
    quick = "--quick" in sys.argv
    max_workers = _parse_workers(sys.argv)
    n_seeds = 3 if quick else params.EVAL_N_SEEDS
    n_sens_seeds = 2 if quick else params.EVAL_SENSITIVITY_N_SEEDS
    resource_counts = [3, 10] if quick else params.EVAL_RESOURCE_COUNTS

    seeds = [params.EVAL_BASE_SEED + i for i in range(n_seeds)]
    sens_seeds = [params.EVAL_BASE_SEED + 500 + i for i in range(n_sens_seeds)]

    checkpoint_path = ROOT / "output" / ("eval_checkpoint_quick.json" if quick else "eval_checkpoint_full.json")
    checkpoint = load_checkpoint(checkpoint_path)

    t_start = time.time()
    print(f"=== Phase 7 evaluation starting === quick={quick} n_seeds={n_seeds} "
          f"sensitivity_seeds={n_sens_seeds} resource_counts={resource_counts} max_workers={max_workers}", flush=True)

    ablation = run_ablation(seeds, params.N_RESOURCES_STUB, checkpoint, checkpoint_path, max_workers)
    sensitivity = run_sensitivity(sens_seeds, resource_counts, checkpoint, checkpoint_path, max_workers)

    elapsed = time.time() - t_start
    print(f"\n=== Evaluation complete in {elapsed / 60:.1f} minutes (this process; checkpointed conditions from a prior run may have taken longer) ===")

    print("\n=== Ablation summary (rescued people, mean +/- std) ===")
    for name, data in ablation.items():
        print(f"  {name:26s}: {data['rescued_people_mean']:6.2f} +/- {data['rescued_people_std']:5.2f}")

    print("\n=== Sensitivity summary ===")
    for strat_name, per_count in sensitivity.items():
        print(f"  {strat_name}:")
        for n_resources in resource_counts:
            data = per_count[str(n_resources)]
            print(f"    {n_resources:3d} resources: {data['rescued_people_mean']:6.2f} +/- {data['rescued_people_std']:5.2f}")

    results = {
        "config": {
            "n_seeds": n_seeds,
            "eval_base_seed": params.EVAL_BASE_SEED,
            "sensitivity_n_seeds": n_sens_seeds,
            "resource_counts": resource_counts,
            "n_resources_default": params.N_RESOURCES_STUB,
            "T_TOTAL_SECONDS": params.T_TOTAL_SECONDS,
            "quick_mode": quick,
            "max_workers": max_workers,
        },
        "ablation": ablation,
        "sensitivity": sensitivity,
        "elapsed_seconds": elapsed,
    }
    out_json = ROOT / "output" / "results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {out_json}")

    plot_ablation(ablation, n_seeds, ROOT / "output" / "charts" / "ablation.png")
    plot_sensitivity(sensitivity, resource_counts, n_sens_seeds, ROOT / "output" / "charts" / "sensitivity.png")
    print(f"Charts written to: {ROOT / 'output' / 'charts'}")


if __name__ == "__main__":
    main()
