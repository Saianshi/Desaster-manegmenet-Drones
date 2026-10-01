## ⚠️ License

© 2026 Saianshi Mohapatra. All rights reserved.

This repository and its code are here only for viewing. You are not allowed to copy, download, reuse or share any part of it without my permission.

If you'd like to use this project or any part of it, please contact me at saianshimohapatraofficial@gmail.com.

# Disaster Response Decision Engine

A decision-support engine for earthquake + urban flash-flood response, built for a hackathon
around one hard constraint: **all comms infrastructure is destroyed**. Drones are the only
sensors, backhaul is unreliable, and every detection is noisy. The engine never sees ground
truth, only what its own sensors and fusion logic can reconstruct.

## The architecture rule

This repo is split into two halves with a hard, enforced boundary:

- **`/engine`**, the actual product. Pure function of `EngineInput -> EngineOutput`
  (see `engine/contracts.py`). It is deployable against real sensor data tomorrow. It has
  **never once imported anything from `/simulation`**, verified programmatically with an AST
  scan after every phase of this build, not just by convention.
- **`/simulation`**, the testbed. Generates a fake earthquake+flood disaster, fake victims,
  fake noisy sensor readings, and advances time. It's allowed to import `/engine` (it has to,
  to speak the same wire format) but the engine can never reach back into it.

Everything the engine "believes" is inferred from noisy `Detection` objects, never from a
`Victim`'s true position, injury severity, or building elevation. Where the engine has to guess
something it fundamentally cannot observe (true injury severity, true elevation, true group
size), that's documented explicitly in `config/params.py` under **ENGINE-SIDE MODEL
ASSUMPTIONS**, separate from the simulation's own ground-truth constants.

## Project layout

```
engine/         The product: contracts, belief fusion, survival models, the triage
                scheduler (the core), drone orchestrator, connectivity inference.
simulation/     The testbed: world/terrain/flood generation, victims, sensors (thermal/
                uwb/rf), the 3-layer comms model, rescue outcome resolution, the main
                run loop, and the phaseN_checkpoint.py scripts used to validate each
                phase of the build.
baseline/       Two comparison strategies implementing the SAME EngineInput->EngineOutput
                interface: nearest_first (no smarts at all) and informed_operator (real
                belief fusion + correct resource-type matching, but no survival scoring,
                cascade bonus, rescuer risk, or connectivity-aware routing).
evaluation/     compare.py runs the full engine against both baselines across many
                seeds, an ablation study (each engine feature on/off), and a
                resource-scarcity sensitivity sweep, parallelised across seeds.
viz/            Presentation layer: side-by-side animation, a static "first responder"
                ops dashboard, a human-readable decision trace, and the final charts.
                Reads only output/*.json, never imports /engine or /simulation (one
                narrow, documented exception in charts.py for the no-rescue mortality
                curve, which is inherently a fresh-simulation question).
config/         Every tunable constant, each tagged [ASSUMPTION] / [PHYSICAL] / [DESIGN]
                with a note on where it came from.
output/         Generated artifacts (charts, logs, the animation, the dashboard).
```

## Quickstart

```bash
pip install -r requirements.txt

# Run the full pipeline once (world -> sensors -> comms -> engine -> rescue resolution)
python -m simulation.runner

# Run any phase's standalone validation checkpoint, e.g.:
python -m simulation.phase2_checkpoint    # mortality curves
python -m simulation.phase6_checkpoint    # orchestrator vs lawnmower coverage

# Run the full evaluation (ablation + sensitivity sweep, parallelised across CPU cores)
python -m evaluation.compare              # add --quick for a fast smoke test

# Build the visualisation layer (needs the DUAL-strategy log, not runner.py's)
python -m simulation.generate_viz_log
python -m viz.animate
python -m viz.responder_view
python -m viz.decision_trace
python -m viz.charts
```

Note: `simulation/runner.py` and `simulation/generate_viz_log.py` both write to
`output/run_log.json`, but in **different schemas** (single-strategy vs dual-strategy), so run
whichever one matches what you're about to do next.

## Key findings (100-seed evaluation, `output/results.json`)

This project's evaluation went through an honest correction cycle worth summarizing, because the
process matters as much as the final numbers:

1. **First evaluation: the engine lost to a simpler baseline.** An "informed operator" (real
   belief fusion + correct resource-type matching, but no survival scoring, cascade bonus, or
   connectivity-aware routing) out-rescued the full engine on average (56.9 vs 49.0 people).
2. **Diagnosed instead of tuned.** Instrumenting every dispatched mission (`evaluation/diagnose_gap.py`)
   showed **>97% were failing**: a fixed 50m search radius routinely missed victims whose
   RF-derived position estimate was 40-60m off, and wrong-resource-type dispatches (a boat sent
   to a buried victim) could repeat forever with nothing learned. At that failure rate, *which*
   target you pick barely matters, raw dispatch volume dominates, which is exactly why the
   faster, dumber baseline was winning.
3. **Two realism fixes to mission resolution** (`simulation/rescue.py`, `engine/models.py`,
   `engine/belief.py`), justified as "what real responders do," not as tuning: search radius now
   scales with the belief's own reported uncertainty and costs real time proportional to the
   area searched (`engine.models.search_minutes`); a wrong-type failure now generates a field
   report (the crew radios back what they actually found) which `BeliefTracker` treats as
   authoritative, overriding remote-sensor classification.
4. **Result:** per-mission success rate rose **5.6x** for the full engine (1.6% to 8.9%). The
   engine/baseline gap closed to statistical noise (paired t=+0.02 at n=100, a 47/47 split across
   seeds). Re-ran the ablation at 100 seeds (up from 30) specifically to stress-test the two
   scoring features that looked promising at 30 seeds:
   - **Cascade bonus: the n=30 signal (t=+1.60) did not survive more data.** t dropped to +0.26 at
     n=100. Reported as the genuine null result it is, not the more flattering smaller-sample
     number.
   - **Rescuer risk moved closer to significance** (t=+1.54 to t=+1.92, just short of the
     conventional 1.96 threshold), borderline/suggestive, not conclusively proven.
   - **Silence-as-signal still shows no effect** (t=-1.14).
- The Phase 6 orchestrator reaches ~97% scarcity-weighted sensor coverage by hour 5 of an 8h
  run; a fixed lawnmower sweep plateaus at ~32% after one pass and never improves again.

Full ablation and sensitivity charts: `output/charts/ablation.png`,
`output/charts/sensitivity.png`. Instrumented mission-outcome diagnostic:
`evaluation/diagnose_gap.py`. Decision-by-decision trace with the engine's own comparative
reasoning: `output/decision_trace.md`.

## Requirements

Python 3, numpy + matplotlib only for the core project. `viz/animate.py` uses `ffmpeg` if
present (falls back to an animated GIF, and says so, if not).
