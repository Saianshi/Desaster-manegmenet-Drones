# Central config for every "invented" number in this project.
# Nothing in /engine or /simulation should hardcode a tunable constant
# inline -- it goes here, with a note on where it came from:
#   [ASSUMPTION] = we made this up for the hackathon, no source
#   [PHYSICAL]   = derived from a real physical/engineering fact
#   [DESIGN]     = arbitrary but intentional design choice (e.g. grid size)

# ---------------------------------------------------------------------------
# World geometry
# ---------------------------------------------------------------------------
MAP_SIZE_M = 2000.0          # [DESIGN] 2km x 2km disaster area, per spec
CELL_SIZE_M = 10.0           # [DESIGN] 10m grid cells, per spec
GRID_N = int(MAP_SIZE_M // CELL_SIZE_M)   # cells per side

# ---------------------------------------------------------------------------
# Population / victims
# ---------------------------------------------------------------------------
N_VICTIMS_STUB = 20          # Phase 1 stub victim count
N_VICTIMS_FULL = 180         # Phase 2+ full scenario victim count
N_BUILDINGS = 200            # [DESIGN] per spec
COLLAPSED_FRACTION = 0.30    # [ASSUMPTION] fraction of buildings collapsed

FRAC_BURIED = 0.40           # [ASSUMPTION] fraction of victims buried in rubble
FRAC_ROOFTOP = 0.40          # [ASSUMPTION] fraction stranded on rooftops in flood zone
FRAC_STREET = 0.20           # [ASSUMPTION] fraction at street level

FRAC_PHONE_ALIVE = 0.70      # [ASSUMPTION] phone on & battery available
FRAC_PHONE_DEAD_BATTERY = 0.20
FRAC_PHONE_NO_APP = 0.10

# ---------------------------------------------------------------------------
# Flood
# ---------------------------------------------------------------------------
DEFAULT_RISE_RATE_M_PER_HR = 0.4   # [ASSUMPTION] flash-flood rise rate, metres/hour

# ---------------------------------------------------------------------------
# Sim clock
# ---------------------------------------------------------------------------
DT_SECONDS = 30.0            # [DESIGN] simulation timestep, seconds
T_TOTAL_SECONDS = 3600.0 * 8 # [DESIGN] simulate 8 hours of response.
                              # Raised from 4->8hr: buried extraction alone
                              # takes 45-90 min, so a 4hr window only fits
                              # 3-4 debris rescues per excavator -- too few
                              # timesteps for the scheduler's trade-offs to
                              # show up in aggregate results.

# ---------------------------------------------------------------------------
# Drones / resources (stub counts, phase 1)
# ---------------------------------------------------------------------------
N_DRONES_STUB = 4
N_RESOURCES_STUB = 5

RANDOM_SEED = 42             # reproducibility

# ---------------------------------------------------------------------------
# Terrain / elevation (Phase 2)
# ---------------------------------------------------------------------------
ELEVATION_MAX_M = 10.0        # [ASSUMPTION] arbitrary elevation scale for the
                               # simulated terrain, 0-10m across the map
ELEVATION_SMOOTH_ITERS = 3    # [DESIGN] box-blur passes to make terrain smooth
ELEVATION_SMOOTH_KERNEL = 21  # [DESIGN] box-blur kernel width in cells (210m)
FLOOD_ZONE_PERCENTILE = 0.30  # [ASSUMPTION] lowest 30% of cells by elevation
                               # are "low-lying" / flood-prone

INITIAL_WATER_LEVEL_M = 4.0   # [ASSUMPTION] the flash flood has already
                               # surged by the time response begins; water
                               # keeps rising from here at DEFAULT_RISE_RATE_M_PER_HR.
                               # Calibrated (with FLOOD_ZONE_RELIEF_M below) so
                               # that over an 8h response window the water
                               # reaches low/short buildings within hours but
                               # not tall ones -- if the initial level is too
                               # low relative to building-height offsets,
                               # nobody in the flood zone ever gets reached.

FLOOD_ZONE_RELIEF_M = 1.5     # [ASSUMPTION] flood zones are, almost by
                               # definition, low and nearly flat -- so a
                               # rooftop victim's GROUND elevation within the
                               # flood zone is rescaled from the raw terrain
                               # map (0-10m, mostly describing the dry
                               # interior) down to this small relief band,
                               # rather than used directly. Building height
                               # (floors * FLOOR_HEIGHT_M) then dominates how
                               # exposed a given rooftop is.

# ---------------------------------------------------------------------------
# Buildings (Phase 2)
# ---------------------------------------------------------------------------
BUILDING_MIN_SIZE_CELLS = 2    # [DESIGN] min building footprint, 20m
BUILDING_MAX_SIZE_CELLS = 6    # [DESIGN] max building footprint, 60m
BUILDING_MIN_FLOORS = 1
BUILDING_MAX_FLOORS = 4        # [ASSUMPTION] low-rise urban/peri-urban area
FLOOR_HEIGHT_M = 2.5           # [PHYSICAL] typical inter-storey height
BUILDING_PLACEMENT_MAX_TRIES = 20  # retries before giving up on overlap

# ---------------------------------------------------------------------------
# Pre-disaster population density prior (Phase 2)
# ---------------------------------------------------------------------------
PRIOR_BASELINE_WEIGHT = 0.15   # [ASSUMPTION] ambient population density
                               # everywhere (streets, open ground)
PRIOR_BUILDING_WEIGHT = 1.0    # [ASSUMPTION] density contribution scale
                               # per building floor
PRIOR_BUILDING_SIGMA_CELLS = 1.5  # [ASSUMPTION] spatial spread of a
                               # building's population "bump", in cells

# ---------------------------------------------------------------------------
# Victim clustering (Phase 2)
# ---------------------------------------------------------------------------
# Group ("cluster") sizes range 1-12 people, weighted toward small
# households with an occasional large group (an apartment floor, a
# classroom, a queue at a bus stop). [ASSUMPTION] weights chosen to give
# wide variance so the scheduler faces genuine trade-offs (e.g. "8 people,
# 40 min window" vs "2 people, 10 min window"), not uniform group sizes.
GROUP_SIZE_WEIGHTS = [30, 20, 13, 10, 8, 6, 5, 4, 3, 3, 2, 2]  # sizes 1..12
GROUP_JITTER_SIGMA_M = 8.0     # [ASSUMPTION] spatial spread of a group
                               # around its site (building or street corner)
N_STREET_CLUSTER_SITES = 7     # [ASSUMPTION] number of street-level
                               # gathering points (e.g. intersections)

# ---------------------------------------------------------------------------
# Battery drain (Phase 2)
# ---------------------------------------------------------------------------
BATTERY_DRAIN_PCT_PER_HOUR_MIN = 6.0   # [ASSUMPTION] phone battery drain
BATTERY_DRAIN_PCT_PER_HOUR_MAX = 14.0  # range, screen-on searching for signal
                                        # drains faster than idle

# ---------------------------------------------------------------------------
# Survival curves (Phase 2, ground truth in simulation)
# ---------------------------------------------------------------------------
# Buried (crush syndrome): survival modelled as an exponential hazard,
# i.e. constant per-hour probability of death, whose rate depends on
# injury_severity. median_survival_hours = time at which 50% of victims
# with that severity have died if never rescued.
BURIED_MEDIAN_SURVIVAL_HOURS_BASE = 6.0     # [ASSUMPTION] median survival,
                                              # injury_severity = 0 (mild)
BURIED_SEVERITY_SURVIVAL_PENALTY = 0.65     # [ASSUMPTION] fraction cut from
                                              # median survival at severity=1
                                              # (severe crush injury)

# Rooftop/flood: victims are safe until the rising water surface comes
# within SUBMERSION_BUFFER_M of their elevation, then face a short, steep
# hazard window (drowning risk), not a slow decline.
SUBMERSION_BUFFER_M = 0.3                    # [ASSUMPTION] margin before
                                              # "water has reached them"
ROOFTOP_MEDIAN_SURVIVAL_MIN_AFTER_SUBMERSION = 10.0  # [ASSUMPTION] minutes

# ---------------------------------------------------------------------------
# Sensor models (Phase 3)
# ---------------------------------------------------------------------------
THERMAL_RANGE_M = 150.0        # [DESIGN] per spec: thermal-IR effective range
THERMAL_CONFIDENCE = 0.85      # [DESIGN] per spec
THERMAL_SIGMA_M = 8.0          # [DESIGN] per spec
THERMAL_MISS_PROB = 0.10       # [ASSUMPTION] chance of a miss despite being
                                # in range (angle, clutter, false negative)

UWB_RANGE_M = 60.0             # [DESIGN] per spec: short-range through-rubble sensor
UWB_CONFIDENCE = 0.6           # [DESIGN] per spec
UWB_SIGMA_M = 15.0             # [DESIGN] per spec
UWB_MISS_PROB = 0.25           # [ASSUMPTION] rubble occlusion even within range
# UWB is "slow": [ASSUMPTION/SIMPLIFICATION] modelled as a single 60m-radius
# scan available PER TIMESTEP ACROSS THE WHOLE FLEET (not per drone) --
# only one uwb-equipped drone actually gets to run its uwb sensor each
# timestep, round-robin. This is what makes "where do we point the one
# ground-penetrating sensor we have" a real orchestration decision later
# (Phase 6), rather than every drone getting free buried-victim detection.

RF_RANGE_M = 300.0             # [DESIGN] per spec
RF_CONFIDENCE = 0.5            # [DESIGN] per spec
RF_SIGMA_M = 40.0              # [DESIGN] per spec
RF_MISS_PROB = 0.15            # [ASSUMPTION]

FALSE_POSITIVE_PROB_PER_SCAN = 0.03  # [ASSUMPTION] chance a given active
                                # (drone, sensor) pair emits one spurious
                                # detection in a timestep, at reduced confidence

# ---------------------------------------------------------------------------
# Comms: three-layer network (Phase 3)
# ---------------------------------------------------------------------------
BLE_RANGE_OPEN_M = 40.0             # [DESIGN] per spec: victim-phone BLE mesh, open air
BLE_RANGE_THROUGH_COLLAPSED_M = 10.0  # [DESIGN] per spec: BLE mesh through a collapsed
                                     # building. [SIMPLIFICATION] applied whenever
                                     # either endpoint of a link is a buried victim,
                                     # rather than raycasting the link path against
                                     # building polygons.
DRONE_CELL_RANGE_M = 300.0          # [DESIGN] per spec: Layer B, drone's portable cell
DRONE_BACKHAUL_RANGE_M = 800.0      # [DESIGN] per spec: Layer C, drone-to-drone backhaul
BASE_STATION_POSITION = (0.0, MAP_SIZE_M / 2.0)  # [DESIGN] fixed base station
                                     # at the map's western edge, per spec

# ---------------------------------------------------------------------------
# ENGINE-SIDE MODEL ASSUMPTIONS (Phase 4)
# ---------------------------------------------------------------------------
# Everything below is used only by /engine. These are assumptions the
# ENGINE has to make because it never sees ground truth -- no true injury
# severity, no true building elevation, no true group size, just noisy
# detections. They intentionally do NOT reference simulation ground
# truth, and they may be *wrong* for any given victim; that mismatch
# between belief and reality is realistic and is exactly why confidence,
# multi-sensor fusion, and hedged classification matter.
#
# The two FRAC_ROOFTOP / FRAC_STREET constants above ARE reused here, but
# only as an aggregate, population-level base rate (the kind of prior a
# real disaster-response planner would have from historical data), never
# as information about any specific victim. That is the same category of
# knowledge as prior_map (pre-disaster public data), not a ground-truth peek.

ENGINE_ASSUMED_INJURY_SEVERITY = 0.5   # [ASSUMPTION] population-average
                                # severity guess for a buried belief --
                                # the engine has no sensor for this at all
ENGINE_ASSUMED_ROOFTOP_CLEARANCE_M = 3.0  # [ASSUMPTION] typical single-storey
                                # clearance above the water level AT FIRST
                                # DETECTION, standing in for the unknown
                                # true building height/elevation
ENGINE_MEDICAL_PENALTY_NO_TEAM = 0.35  # [ASSUMPTION] multiplicative survival
                                # penalty applied at extraction (reperfusion
                                # / crush syndrome injury) if a buried
                                # victim is freed without a medical team present
ENGINE_DEFAULT_SURVIVAL_PROB_UNMODELED = 0.95  # [ASSUMPTION] street/unknown
                                # env_type: no active hazard curve modelled
                                # (matches the simulation's own street-victim
                                # assumption), small residual risk only

ENGINE_BELIEF_CONFIDENCE_DECAY_PER_MIN = 0.03  # [ASSUMPTION] confidence lost
                                # per minute a belief goes without a
                                # re-detection
ENGINE_BELIEF_PRUNE_THRESHOLD = 0.05   # [ASSUMPTION] drop a belief once its
                                # confidence decays below this
ENGINE_FUSION_SIGMA_MULTIPLIER = 2.0   # [DESIGN] per spec: detections within
                                # ~2 sigma are treated as the same cluster

ENGINE_GROUP_SIZE_DETECTIONS_PER_STEP = 4  # [ASSUMPTION] heuristic: every
                                # N independent re-detections of the same
                                # cluster nudges the group-size guess up by
                                # one (more phones/heat signatures re-firing
                                # is weak evidence of more people) -- the
                                # engine has no direct way to count people
ENGINE_GROUP_SIZE_MAX_GUESS = 8            # [ASSUMPTION] cap on that guess;
                                # deliberately below the true max (12) since
                                # this is a weak signal

ENGINE_SILENT_ZONE_PRIOR_PERCENTILE = 60.0  # [ASSUMPTION] a cell counts as
                                # "high prior population" if it's in the
                                # top 40% of this run's prior_map values
ENGINE_SILENT_ZONE_MIN_RISK = 0.3   # [ASSUMPTION] minimum combined risk
                                # score (prior x scan-confidence) to report
                                # a cell as a suspected silent zone
ENGINE_SILENT_ZONE_MAX_REPORTED = 25  # [DESIGN] cap on how many zones are
                                # surfaced per decision -- an operator
                                # needs the top handful, not every one of
                                # potentially thousands of qualifying
                                # 10m grid cells once scanning has covered
                                # much of a high-prior area
ENGINE_SILENT_ZONE_MIN_SEPARATION_M = 50.0  # [DESIGN] minimum distance
                                # between reported zone centres -- without
                                # this, one large contiguous unscanned
                                # blob reports as many adjacent 10m cells
                                # instead of one distinct zone

# Extraction time estimates (Phase 4). [DESIGN] values are the spec's own
# "boat rescue ~8 min; debris extraction 45-90 min scaled by group_size".
EXTRACTION_BOAT_MINUTES = 8.0
DEBRIS_EXTRACTION_BASE_MIN = 45.0   # group_size == 1
DEBRIS_EXTRACTION_MAX_MIN = 90.0    # group_size >= 12 (reference "large group")
STREET_EXTRACTION_MINUTES = 5.0     # [ASSUMPTION] not specified; simple
                                     # evacuation assistance, no specialised
                                     # equipment needed

# ---------------------------------------------------------------------------
# Triage scheduler (Phase 5, engine/scheduler.py)
# ---------------------------------------------------------------------------
# CASCADE BONUS: rescuing a live-phone cluster near unexplored/unconnected
# territory is worth more than the rescue alone, because getting a
# resource (and its drone escort) out there extends network reach.
ENGINE_CASCADE_RADIUS_M = DRONE_CELL_RANGE_M  # [DESIGN] reuse the drone's own
                                # cell-pickup range: this is literally how
                                # far being physically present there would
                                # extend coverage
ENGINE_CASCADE_BONUS_MAX = 5.0  # [ASSUMPTION] value ceiling (in the same
                                # units as base_value, i.e. "equivalent
                                # lives") for fully unlocking a totally
                                # silent, fully-unscanned high-population
                                # pocket -- scaled against a LOCAL density
                                # yardstick (see _local_unexplored_scale in
                                # scheduler.py), not the whole map's total
                                # population. [FIXED, Phase 7 problem 2a]:
                                # normalising against the full 2km map made
                                # every local contribution vanishingly
                                # small by construction (max observed value
                                # was 0.177 against a typical base_value
                                # near 5) -- diagnosed via
                                # engine/scheduler.py's own instrumentation,
                                # not tuned to produce a bigger number.
ENGINE_CASCADE_DISCONNECTED_BONUS = 2.5  # [ASSUMPTION] flat bonus (half of
                                # ENGINE_CASCADE_BONUS_MAX) for trigger B:
                                # this belief sits near a drone that
                                # currently has no multi-hop backhaul path
                                # to base (see engine/connectivity.py).
                                # Added in Phase 7 problem 2b as a SECOND,
                                # longer-lived cascade trigger -- trigger A
                                # (above) empties out once the Phase 6
                                # orchestrator's exploration approaches full
                                # coverage (~97% by hour 5); a drone losing
                                # backhaul as it ranges out can happen at
                                # any point across the whole run, so this
                                # doesn't decay the way unexplored area does.

# RESCUER RISK PENALTY: subtracted from value. Flood risk scales with how
# fast the water is rising (a slow rise is a minor inconvenience for a
# boat crew; a fast one is dangerous). Buried/collapsed-structure risk is
# a flat elevated constant -- env_type == "buried" already IS the signal
# that a structure has collapsed there.
RESCUER_RISK_BOAT_BASE = 0.3     # [ASSUMPTION]
RESCUER_RISK_WATER_COEFF = 0.5   # [ASSUMPTION] extra risk per m/hr of rise_rate
RESCUER_RISK_BURIED_BASE = 0.8   # [ASSUMPTION] unstable/collapsed structure danger
RESCUER_RISK_STREET_BASE = 0.05  # [ASSUMPTION] minimal

# ---------------------------------------------------------------------------
# Rescue outcome resolution (Phase 5, simulation/rescue.py)
# ---------------------------------------------------------------------------
# This is ground-truth-side machinery: given a resource physically sent to
# an ESTIMATED belief position, what real victim (if any) does it find?
#
# [REALISM FIX, post-Phase-7 diagnostic] A fixed 50m search radius against
# RF-derived beliefs (position_sigma up to 40m) was the single largest
# cause of mission failure (73% of full_engine's dispatches). Real teams
# don't search a fixed disc regardless of how uncertain the fix is -- they
# scale the search to the reported uncertainty, and it costs them time to
# do it. Both effects are modelled below.
RESCUE_SEARCH_RADIUS_UNCERTAINTY_MULTIPLIER = 2.0  # [DESIGN] search out to
                                # roughly 2x the belief's own reported
                                # position uncertainty
RESCUE_SEARCH_RADIUS_FLOOR_M = 50.0   # [DESIGN] never search a smaller
                                # radius than this even for a very tight fix
RESCUE_SEARCH_RADIUS_CAP_M = 150.0    # [DESIGN] never search wider than
                                # this -- beyond it a team calls it a false
                                # lead and moves on rather than expanding
                                # the search indefinitely
SEARCH_TIME_AT_FLOOR_MIN = 5.0  # [ASSUMPTION] minutes to thoroughly sweep
                                # the floor-radius (50m) area on scene.
                                # Search time scales with AREA from there
                                # (radius^2), not radius, so a vague belief
                                # costs meaningfully more responder time --
                                # not just a wider miss chance -- than a
                                # well-localised one. At the 150m cap this
                                # is 45 min (5 * (150/50)^2), vs 5 min at
                                # the floor.

PARTNER_WAIT_TIMEOUT_MIN = 30.0  # [ASSUMPTION] a buried rescue needs
                                # excavator + medical simultaneously; if
                                # the second resource never shows up
                                # within this long, the first gives up
                                # rather than sitting idle forever

# [REALISM FIX 2] On a wrong_resource_type failure, the crew is physically
# standing at the site and radios back what they actually found -- real
# responders don't just walk away and let the same wrong dispatch repeat
# forever. This is what that radio report looks like as a Detection:
# small position_sigma ("the crew is standing there"), high confidence
# (direct human observation, not a probabilistic remote sensor read).
FIELD_REPORT_SIGMA_M = 3.0      # [ASSUMPTION]
FIELD_REPORT_CONFIDENCE = 0.97  # [ASSUMPTION] near-certain, direct observation

# ---------------------------------------------------------------------------
# Drone orchestration (Phase 6, engine/orchestrator.py)
# ---------------------------------------------------------------------------
# Drone.speed isn't in the contract (unlike Resource), so this is the
# shared assumption for how far a drone can reasonably travel in one
# timestep -- a drone's own cruise speed is public hardware spec, fair
# for the engine to know (not victim ground truth).
DRONE_SPEED_M_PER_MIN = 400.0    # [ASSUMPTION] ~24 km/h, small quadcopter cruise

ORCHESTRATOR_N_ANGLES = 6        # [DESIGN] candidate waypoints: compass
                                # directions at 2 step radii, plus "stay put"
                                # (trimmed from 8 during Phase 7 performance
                                # tuning -- the sweep needs O(100) full runs)
ORCHESTRATOR_SILENT_ZONE_WEIGHT = 3.0  # [ASSUMPTION] extra pull toward a
                                # FLAGGED suspected silent zone, beyond its
                                # raw prior-population contribution to
                                # information_gain -- actively investigate
                                # flagged risk, don't just wander to any
                                # unscanned cell
ORCHESTRATOR_CONNECTIVITY_PENALTY = 60.0  # [ASSUMPTION] "LARGE" relative to
                                # a typical single-candidate information_gain
                                # (tens, occasionally low hundreds for a
                                # rich unscanned area) -- large enough that
                                # a marginal information_gain difference
                                # never justifies breaking the chain, but
                                # NOT effectively infinite: a genuinely
                                # rich unexplored area can still outweigh
                                # it. An absolute veto (first tried: 1e6)
                                # caused a worse failure mode -- every
                                # drone stayed paranoid about breaking ITS
                                # OWN link and the whole fleet clustered
                                # near base instead of fanning out into a
                                # connected chain, tanking coverage. A
                                # finite-but-large penalty is also more
                                # consistent with the fact that a
                                # disconnected drone's data is BUFFERED,
                                # not lost (comms.CommsRelay) -- temporary
                                # disconnection to search a high-value area
                                # is a real, recoverable trade-off, not a
                                # catastrophe.
ORCHESTRATOR_TRAVEL_COST_PER_M = 0.01  # [ASSUMPTION] small per-metre cost --
                                # acts as a tie-breaker among similar-gain
                                # candidates, not a dominant term
ORCHESTRATOR_LOW_BATTERY_PCT = 20.0        # [ASSUMPTION]
ORCHESTRATOR_LOW_BATTERY_COST_MULTIPLIER = 3.0  # [ASSUMPTION] drone battery
                                # is static (100%) in this simulation, so
                                # this branch is currently dormant -- wired
                                # for if/when drone battery drain is added

# information_gain is computed PER SENSOR the drone carries, each using
# that sensor's own range and own "have I scanned this cell" history
# (see BeliefTracker's per-sensor scan tracking). Weights make uwb
# coverage worth much more than rf: rf's 300m passive range sweeps huge
# areas in the first few timesteps regardless of whether it finds anyone,
# while uwb -- the ONLY sensor that can find buried victims -- only ever
# covers a 60m circle, once per timestep, fleet-wide. Without this
# weighting, the orchestrator "explores" (via cheap rf coverage) almost
# the whole map in the first few minutes and then has no signal left to
# steer by for the rest of an 8h run.
ORCHESTRATOR_SENSOR_INFO_WEIGHT = {
    "uwb": 5.0,      # [ASSUMPTION] scarce, high-value: the only way to find buried victims
    "thermal": 1.0,  # [ASSUMPTION] moderate range, moderate value
    "rf": 0.3,       # [ASSUMPTION] huge range, "cheap" -- gets covered incidentally
}

# ---------------------------------------------------------------------------
# Naive baseline (Phase 7, baseline/nearest_first.py)
# ---------------------------------------------------------------------------
BASELINE_TARGET_MERGE_RADIUS_M = 30.0  # [ASSUMPTION] a human operator
                                # watching a screen mentally groups two
                                # blips this close together into one marker
                                # -- the baseline's only "fusion", vs the
                                # real engine's confidence/sensor-aware fusion

# ---------------------------------------------------------------------------
# Evaluation (Phase 7, evaluation/compare.py)
# ---------------------------------------------------------------------------
EVAL_N_SEEDS = 100              # [DESIGN] raised from 30 -- at n=30 the cascade
                                # bonus (t=1.60) and rescuer risk (t=1.54) paired
                                # ablation effects were suggestive but short of
                                # conventional significance (t~1.96); n=100
                                # narrows the standard error enough to tell
                                # whether they're real or just didn't have the
                                # sample size to clear the bar
EVAL_BASE_SEED = 1000          # [DESIGN] offset from RANDOM_SEED so eval
                                # seeds never collide with the demo seed
EVAL_RESOURCE_COUNTS = [3, 5, 10, 20, 40]  # [ASSUMPTION] scarce -> plentiful,
                                # sweeping resource count for the sensitivity
                                # analysis (multiples of 3 keep the
                                # boat/excavator/medical type-cycle even)
EVAL_SENSITIVITY_N_SEEDS = 10   # [ASSUMPTION] fewer seeds than the main
                                # comparison -- the sensitivity sweep has
                                # EVAL_RESOURCE_COUNTS x 2 strategies x this
                                # many runs already, and it's asking a
                                # coarser question (does the GAP trend hold)
                                # than the headline mean/std comparison
