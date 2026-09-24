# Decision Trace -- seed 42

Scenario: 189 people across 55 victim groups, 8-hour response window.

**Outcome**: engine rescued **63** people (13 missions); baseline (nearest-first) rescued **0** people (0 missions).

> Note: this specific seed is an extreme case for the baseline. Across the full 30-seed evaluation (Phase 7) the baseline's mean was around 4-5 people rescued, not literally zero -- zero here reflects this one run, not the general finding.

## Engine rescue timeline

**T+00:54** -- rescued 1 person (rooftop) via r3.
> Boat r3 -> belief_78 (384 min window, 0 min travel, value=6.53) over farther belief_33 (369 min window, value=6.51); belief_33 would no longer be reachable this round.

**T+02:06** -- rescued 8 people (buried) via r1.
> Excavator r1 + Medical r2 -> belief_236 (243 min window, 5 min travel, value=4.63) over farther belief_173 (243 min window, value=3.41); belief_173 would no longer be reachable this round.

**T+02:07** -- rescued 2 people (rooftop) via r3.
> Boat r3 -> belief_248 (400 min window, 8 min travel, value=2.99) over nearer belief_204 (372 min window, value=2.73); belief_204 would no longer be reachable this round.

**T+02:25** -- rescued 4 people (rooftop) via r0.
> Boat r0 -> belief_248 (382 min window, 0 min travel, value=7.14) over farther belief_274 (385 min window, value=5.37); belief_274 would no longer be reachable this round.

**T+02:39** -- rescued 1 person (rooftop) via r0.
> Boat r0 -> belief_274 (371 min window, 7 min travel, value=3.34) over nearer belief_237 (368 min window, value=3.15); belief_237 would no longer be reachable this round.

**T+03:11** -- rescued 6 people (rooftop) via r0.
> Boat r0 -> belief_327 (384 min window, 0 min travel, value=3.82) over farther belief_273 (339 min window, value=3.30); belief_273 would no longer be reachable this round.

**T+03:21** -- rescued 6 people (buried) via r1.
> Excavator r1 + Medical r2 -> belief_236 (243 min window, 5 min travel, value=4.63) over farther belief_173 (243 min window, value=3.41); belief_173 would no longer be reachable this round.

**T+03:24** -- rescued 5 people (rooftop) via r0.
> Boat r0 -> belief_321 (369 min window, 2 min travel, value=3.16) over farther belief_375 (411 min window, value=2.23); belief_375 would no longer be reachable this round.

**T+03:37** -- rescued 12 people (street) via r2.
> Medical r2 -> belief_281 (no deadline, 0 min travel, value=5.96) over nearer belief_293 (no deadline, value=5.74); belief_293 would no longer be reachable this round.

**T+03:39** -- rescued 6 people (rooftop) via r0.
> Boat r0 -> belief_376 (396 min window, 6 min travel, value=4.06) over nearer belief_375 (396 min window, value=3.57); belief_375 would no longer be reachable this round.

**T+04:07** -- rescued 1 person (rooftop) via r0.
> Boat r0 -> belief_395 (386 min window, 4 min travel, value=2.82) over farther belief_413 (412 min window, value=0.82); belief_413 would no longer be reachable this round.

**T+04:07** -- rescued 4 people (street) via r2.
> Medical r2 -> belief_376 (no deadline, 0 min travel, value=3.64) over farther belief_397 (no deadline, value=0.90); belief_397 would no longer be reachable this round.

**T+05:54** -- rescued 7 people (buried) via r4.
> Excavator r4 + Medical r2 -> belief_413 (243 min window, 5 min travel, value=3.15) -- no comparable alternative for this resource this round.

## Possible cascade events

_A rescue is flagged here if, within 30 minutes afterward, one or more previously-undetected victims were first detected within 300m of the rescue site. This is a proximity/timing heuristic, not a proven causal link -- a drone could simply have been passing through anyway._

_No cascade-pattern events detected this run._

## Baseline attempts (for contrast)

Baseline made 137 total mission attempts, 0 successful (0.0%).
