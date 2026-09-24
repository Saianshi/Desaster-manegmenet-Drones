"""
Side-by-side animation: baseline (left) vs our engine (right), same
clock, same seed. Reads output/run_log.json (simulation/generate_viz_log.py's
dual-strategy log) -- does not import /engine or /simulation.

Each panel: building footprints (collapsed = darker), flood extent
(grows with water_level), victims (grey = undetected, colour-coded
red->green by urgency once detected, black X once dead, green circle
once rescued), drones (triangle, green if it currently has backhaul to
base, red if not) with a faint coverage-radius circle, dashed backhaul
links between drones/base, and resource units (square=boat,
plus=excavator, diamond=medical). A clock and a live
"rescued / dead / at risk" counter sit above each panel.

Renders PNG frames to output/frames/{baseline,engine}/frame_NNNN.png,
stitches to MP4 via ffmpeg if available (falls back to an animated GIF
via matplotlib's Pillow writer, which needs no external binary, and
says plainly which one it produced), and exports 4 annotated stills at
full resolution: start, divergence point, mid-run, and end.

Run standalone with:  python -m viz.animate [--stride N] [--fps N]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from viz import logdata

RESOURCE_MARKERS = {"boat": "s", "excavator": "P", "medical": "D"}
SENSOR_RANGE_FOR_COVERAGE = params.RF_RANGE_M  # broadest sensor -- the coverage circle spec asks for

# dark theme, matching docs/index.html and viz/player.html's canvas palette
BG = "#0d1117"
PANEL = "#0b0e14"
GRID = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
    "text.color": TEXT, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
})


def _urgency_color(urgency: float):
    # green (safe) -> red (imminent). RdYlGn reversed so 0=green, 1=red.
    return plt.cm.RdYlGn_r(min(max(urgency, 0.0), 1.0))


def draw_panel(ax, log: dict, run_name: str, frame_idx: int, static: dict) -> dict:
    """Draws one panel for one frame. Returns a small dict of live counts
    (rescued/dead/at_risk people) for the caller to render as text."""
    ax.clear()
    map_size = log["meta"]["map_size_m"]
    entry = log["runs"][run_name]["entries"][frame_idx]
    t = entry["input"]["t"]
    water_level = entry["input"]["environment"]["water_level"]
    gt = log["runs"][run_name]["ground_truth"]["timesteps"][frame_idx]
    rescued_now = static["rescued_by_frame"][run_name][frame_idx]
    first_detected = static["first_detected"][run_name]
    group_size = static["group_size"]

    for b in log["world_static"]["buildings"]:
        color = "#33383f" if b["collapsed"] else "#4d5561"
        ax.add_patch(Rectangle((b["x0"], b["y0"]), b["x1"] - b["x0"], b["y1"] - b["y0"],
                                facecolor=color, edgecolor="none", alpha=0.75, zorder=1))

    flood_mask = static["elevation_map"] <= water_level
    flood_img = np.where(flood_mask, 1.0, np.nan)
    ax.imshow(flood_img, extent=(0, map_size, 0, map_size), origin="lower",
              cmap=ListedColormap(["#2E75C4"]), vmin=0, vmax=1, alpha=0.32, zorder=2)

    n_rescued_people = 0
    n_dead_people = 0
    n_at_risk_people = 0
    for v in gt["victims"]:
        vid = v["victim_id"]
        pos = v["position"]
        gs = group_size.get(vid, 1)
        if vid in rescued_now:
            n_rescued_people += gs
            ax.scatter(*pos, c="#2ecc71", marker="o", s=42, zorder=6, edgecolors=BG, linewidths=0.5)
        elif not v["alive"]:
            n_dead_people += gs
            ax.scatter(*pos, c="#e6edf3", marker="x", s=38, zorder=6, linewidths=1.6)
        else:
            n_at_risk_people += gs
            detected = first_detected.get(vid, float("inf")) <= t
            if detected:
                ax.scatter(*pos, c=[_urgency_color(v.get("urgency", 0.0))], marker="o", s=30,
                           zorder=5, edgecolors=BG, linewidths=0.3)
            else:
                ax.scatter(*pos, c="#8b949e", marker="o", s=16, zorder=3, alpha=0.6)

    drones = entry["input"]["drones"]
    has_backhaul = logdata.compute_backhaul_from_linked_to(drones)
    base_pos = tuple(params.BASE_STATION_POSITION)
    for d in drones:
        pos = d["position"]
        color = "#3fb950" if has_backhaul.get(d["drone_id"], False) else "#f85149"
        ax.add_patch(Circle(pos, SENSOR_RANGE_FOR_COVERAGE, fill=False, edgecolor=color, alpha=0.28, linewidth=0.8, zorder=2))
        ax.scatter(*pos, marker="^", s=75, c=color, edgecolors=BG, linewidths=0.6, zorder=7)
        for nb in d["linked_to"]:
            other_pos = base_pos if nb == "base" else next((x["position"] for x in drones if x["drone_id"] == nb), None)
            if other_pos is not None:
                link_color = "#f85149" if not (has_backhaul.get(d["drone_id"], False) and (nb == "base" or has_backhaul.get(nb, False))) else "#3fb950"
                ax.plot([pos[0], other_pos[0]], [pos[1], other_pos[1]], linestyle="--", linewidth=0.9, color=link_color, alpha=0.8, zorder=4)
    ax.scatter(*base_pos, marker="*", s=220, c="gold", edgecolors=BG, linewidths=0.6, zorder=8)

    for r in entry["input"]["resources"]:
        marker = RESOURCE_MARKERS.get(r["type"], "o")
        ax.scatter(*r["position"], marker=marker, s=55, c="#f1c40f", edgecolors=BG, linewidths=0.5, zorder=7)

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_title(logdata.RUN_LABELS.get(run_name, run_name), fontsize=13, fontweight="bold", color=TEXT)

    return {"rescued": n_rescued_people, "dead": n_dead_people, "at_risk": n_at_risk_people}


def _format_clock(t_seconds: float) -> str:
    h = int(t_seconds // 3600)
    m = int((t_seconds % 3600) // 60)
    return f"T+{h:02d}:{m:02d}"


def render_frame(log: dict, frame_idx: int, static: dict, fig=None, axes=None):
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    counts = {}
    for ax, run_name in zip(axes, ("baseline", "engine")):
        counts[run_name] = draw_panel(ax, log, run_name, frame_idx, static)

    t = logdata.frame_t(log, "engine", frame_idx)
    fig.suptitle(
        f"{_format_clock(t)}   |   Baseline: {counts['baseline']['rescued']} rescued, "
        f"{counts['baseline']['dead']} dead, {counts['baseline']['at_risk']} at risk"
        f"      Engine: {counts['engine']['rescued']} rescued, "
        f"{counts['engine']['dead']} dead, {counts['engine']['at_risk']} at risk",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, axes


def find_divergence_frame(log: dict) -> int:
    """First frame where cumulative people-rescued differs by >= 3
    between engine and baseline -- 'the moment the two strategies
    visibly diverge'. Falls back to the midpoint if they never diverge
    by that much."""
    n = logdata.n_frames(log, "engine")
    group_size = logdata.group_size_lookup(log)
    rescued_e = logdata.rescued_ids_by_frame(log, "engine", n)
    rescued_b = logdata.rescued_ids_by_frame(log, "baseline", n)
    for i in range(n):
        pe = sum(group_size.get(v, 1) for v in rescued_e[i])
        pb = sum(group_size.get(v, 1) for v in rescued_b[i])
        if abs(pe - pb) >= 3:
            return i
    return n // 2


def build_static(log: dict) -> dict:
    n = logdata.n_frames(log, "engine")
    elevation_map = np.array(log["world_static"]["elevation_map"])
    return {
        "elevation_map": elevation_map,
        "group_size": logdata.group_size_lookup(log),
        "first_detected": {
            "engine": logdata.first_detected_times(log, "engine"),
            "baseline": logdata.first_detected_times(log, "baseline"),
        },
        "rescued_by_frame": {
            "engine": logdata.rescued_ids_by_frame(log, "engine", n),
            "baseline": logdata.rescued_ids_by_frame(log, "baseline", n),
        },
    }


def render_all_frames(log: dict, static: dict, stride: int, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = logdata.n_frames(log, "engine")
    frame_indices = list(range(0, n, stride))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    saved_paths = []
    for k, i in enumerate(frame_indices):
        render_frame(log, i, static, fig=fig, axes=axes)
        path = out_dir / f"frame_{k:05d}.png"
        fig.savefig(path, dpi=110, facecolor=BG)
        saved_paths.append(path)
        if k % 20 == 0:
            print(f"  rendered frame {k + 1}/{len(frame_indices)} (t={logdata.frame_t(log, 'engine', i) / 3600:.2f}h)", flush=True)
    plt.close(fig)
    return saved_paths


def stitch_video(frame_paths: list, out_path: Path, fps: int) -> str:
    if not frame_paths:
        return "no frames to stitch"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        frame_dir = frame_paths[0].parent
        pattern = str(frame_dir / "frame_%05d.png")
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
               "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return f"MP4 written to {out_path}"
        return f"ffmpeg failed (exit {result.returncode}): {result.stderr[-500:]}"
    else:
        gif_path = out_path.with_suffix(".gif")
        import matplotlib.animation as animation
        from PIL import Image
        images = [Image.open(p) for p in frame_paths]
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                        duration=int(1000 / fps), loop=0)
        return (f"ffmpeg NOT FOUND on this system -- wrote an animated GIF instead "
                f"({gif_path}). Install ffmpeg and re-run to get an MP4.")


def export_stills(log: dict, static: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = logdata.n_frames(log, "engine")
    divergence_idx = find_divergence_frame(log)
    key_frames = {
        "start": 0,
        "divergence": divergence_idx,
        "midrun": n // 2,
        "end": n - 1,
    }
    for name, idx in key_frames.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        render_frame(log, idx, static, fig=fig, axes=axes)
        path = out_dir / f"still_{name}.png"
        fig.savefig(path, dpi=220, facecolor=BG)
        plt.close(fig)
        print(f"  still '{name}' (frame {idx}, t={logdata.frame_t(log, 'engine', idx) / 3600:.2f}h) -> {path}")


def main():
    stride = 6  # every 3 min of sim time -> 160 frames over an 8h run
    fps = 8     # -> a 20s video
    argv = sys.argv[1:]
    if "--stride" in argv:
        stride = int(argv[argv.index("--stride") + 1])
    if "--fps" in argv:
        fps = int(argv[argv.index("--fps") + 1])

    print("Loading log...")
    log = logdata.load_log()
    print("Precomputing derived data (detection times, rescue timelines)...")
    static = build_static(log)

    frames_dir = ROOT / "output" / "frames"
    print(f"Rendering frames (stride={stride}) to {frames_dir} ...")
    frame_paths = render_all_frames(log, static, stride, frames_dir)
    print(f"Rendered {len(frame_paths)} frames.")

    video_path = ROOT / "output" / "run_animation.mp4"
    status = stitch_video(frame_paths, video_path, fps)
    print(status)

    stills_dir = ROOT / "output" / "charts" / "stills"
    print(f"Exporting key stills to {stills_dir} ...")
    export_stills(log, static, stills_dir)


if __name__ == "__main__":
    main()
