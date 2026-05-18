"""plot_pareto.py — AxMAC Pareto front visualiser.

Put this file next to your axmac/ package directory (i.e. at the same level
as the axmac/ folder, not inside it) and run:

    python plot_pareto.py

It will:
  1. Run the full INT + FP design-space sweep (same knobs used in the project).
  2. Extract Pareto fronts for every error metric.
  3. Save  pareto_energy_vs_nmed.png   (main plot, energy vs NMED)
           pareto_all_metrics.png       (2×2 grid, all four error metrics)
           pareto_front_table.csv       (Pareto-front points, NMED axis)
  4. Also pop up an interactive matplotlib window if a display is available.

Dependencies: matplotlib, numpy (both already required by the project).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the axmac package is importable when running the script directly
# from the repo root.  Adjust the path below if your layout differs.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from .exact_mac import INT4, INT8, INT16, BF16, FP16, FP32
from .pareto import (
    DesignPoint,
    pareto_front,
    sort_front_by_energy,
    sweep_all_designs,
)

# ============================================================
# 1.  Design-space sweep — edit knobs here if needed
# ============================================================

INT_FMTS      = [INT4, INT8, INT16]
FP_FMTS       = [FP16, BF16, FP32]
INT_K_VALUES  = [0, 1, 2, 3, 4]
INT_ACA_WIN   = [None, 16, 8, 4]   # None = exact carry chain
FP_K_VALUES   = [0, 1, 2, 3, 4, 5]
N_SAMPLES     = 1000

print("Running design-space sweep …", flush=True)
all_points: list[DesignPoint] = sweep_all_designs(
    INT_FMTS, FP_FMTS,
    INT_K_VALUES, INT_ACA_WIN, FP_K_VALUES,
    n_samples=N_SAMPLES,
)
print(f"  {len(all_points)} design points collected.")

# ============================================================
# 2.  Colour / marker scheme  (one colour per format)
# ============================================================

FMT_STYLE: dict[str, dict] = {
    "INT4":  {"color": "#185FA5", "marker": "o"},
    "INT8":  {"color": "#378ADD", "marker": "s"},
    "INT16": {"color": "#85B7EB", "marker": "^"},
    "FP16":  {"color": "#0F6E56", "marker": "D"},
    "BF16":  {"color": "#1D9E75", "marker": "P"},
    "FP32":  {"color": "#5DCAA5", "marker": "X"},
}
PARETO_COLOR  = "#D85A30"
PARETO_ZORDER = 5

# ============================================================
# 3.  Helper: draw one Pareto scatter panel
# ============================================================

ERROR_LABELS: dict[str, str] = {
    "error_nmed":    "NMED (normalised median error distance)",
    "error_rmse":    "RMSE",
    "error_med":     "MED (median absolute error)",
    "error_max_abs": "Max absolute error",
}


def _draw_panel(
    ax: plt.Axes,
    points: list[DesignPoint],
    *,
    x_key: str = "energy_pJ",
    y_key: str = "error_nmed",
    annotate_front: bool = True,
) -> None:
    """Draw all design points + Pareto front onto *ax*."""

    front      = pareto_front(points, x_key=x_key, y_key=y_key)
    front_set  = {id(p) for p in front}
    front_sorted = sort_front_by_energy(front)

    # --- non-Pareto points, grouped by format ---
    non_front = [p for p in points if id(p) not in front_set]
    for fmt_name, style in FMT_STYLE.items():
        pts = [p for p in non_front if p.fmt_name == fmt_name]
        if not pts:
            continue
        ax.scatter(
            [p.energy_pJ for p in pts],
            [getattr(p, y_key) for p in pts],
            color=style["color"],
            marker=style["marker"],
            s=40,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )

    # --- Pareto front points ---
    for fmt_name, style in FMT_STYLE.items():
        pts = [p for p in front if p.fmt_name == fmt_name]
        if not pts:
            continue
        ax.scatter(
            [p.energy_pJ for p in pts],
            [getattr(p, y_key) for p in pts],
            color=PARETO_COLOR,
            edgecolors=style["color"],
            marker=style["marker"],
            s=100,
            linewidths=1.8,
            zorder=PARETO_ZORDER,
        )

    # --- Pareto staircase line ---
    if len(front_sorted) > 1:
        fx = [p.energy_pJ        for p in front_sorted]
        fy = [getattr(p, y_key)  for p in front_sorted]
        ax.plot(fx, fy,
                color=PARETO_COLOR, linewidth=1.5,
                linestyle="--", zorder=PARETO_ZORDER - 1,
                label="Pareto front")

    # --- optional text annotations on Pareto points ---
    if annotate_front:
        for p in front_sorted:
            w_str = f"W{p.aca_window}" if p.aca_window else ""
            label = f"{p.fmt_name}\nK={p.K}{(' '+w_str) if w_str else ''}"
            ax.annotate(
                label,
                xy=(p.energy_pJ, getattr(p, y_key)),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=6.5,
                color="#444",
                zorder=PARETO_ZORDER + 1,
            )

    ax.set_xlabel("Energy per MAC  (pJ)", fontsize=10)
    ax.set_ylabel(ERROR_LABELS.get(y_key, y_key), fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# ============================================================
# 4.  Legend helpers
# ============================================================

def _format_legend_handles() -> list[mlines.Line2D]:
    handles = []
    for name, style in FMT_STYLE.items():
        h = mlines.Line2D(
            [], [],
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markersize=7,
            label=name,
        )
        handles.append(h)
    handles.append(mlines.Line2D(
        [], [],
        color=PARETO_COLOR,
        marker="o",
        linestyle="--",
        markersize=8,
        markeredgecolor="#185FA5",
        markeredgewidth=1.5,
        label="Pareto front",
    ))
    return handles


# ============================================================
# 5.  Main plot: energy vs NMED
# ============================================================

print("Generating main plot (energy vs NMED) …", flush=True)

fig_main, ax_main = plt.subplots(figsize=(8, 5))
_draw_panel(ax_main, all_points, y_key="error_nmed", annotate_front=True)
ax_main.set_title(
    "AxMAC design space: energy vs NMED\n"
    "(Pareto-optimal points highlighted, annotated with format / K / ACA-window)",
    fontsize=11,
)
ax_main.legend(
    handles=_format_legend_handles(),
    fontsize=8,
    framealpha=0.9,
    loc="upper right",
)
fig_main.tight_layout()
out_main = _HERE / "pareto_energy_vs_nmed.png"
fig_main.savefig(out_main, dpi=150)
print(f"  Saved → {out_main}")

# ============================================================
# 6.  2×2 grid: all four error metrics
# ============================================================

print("Generating 2×2 grid (all metrics) …", flush=True)

ERROR_KEYS = ["error_nmed", "error_rmse", "error_med", "error_max_abs"]
fig_grid, axes = plt.subplots(2, 2, figsize=(12, 8))
fig_grid.suptitle("AxMAC Pareto fronts — all error metrics", fontsize=13, y=1.01)

for ax, y_key in zip(axes.flat, ERROR_KEYS):
    _draw_panel(ax, all_points, y_key=y_key, annotate_front=False)
    ax.set_title(ERROR_LABELS[y_key], fontsize=9)

# shared legend below the grid
fig_grid.legend(
    handles=_format_legend_handles(),
    loc="lower center",
    ncol=4,
    fontsize=8,
    framealpha=0.9,
    bbox_to_anchor=(0.5, -0.04),
)
fig_grid.tight_layout()
out_grid = _HERE / "pareto_all_metrics.png"
fig_grid.savefig(out_grid, dpi=150, bbox_inches="tight")
print(f"  Saved → {out_grid}")

# ============================================================
# 7.  CSV export of the NMED Pareto front
# ============================================================

front_nmed = sort_front_by_energy(
    pareto_front(all_points, x_key="energy_pJ", y_key="error_nmed")
)

out_csv = _HERE / "pareto_front_table.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "fmt_name", "is_fp", "K", "aca_window",
        "energy_pJ", "mult_pJ", "add_pJ",
        "error_nmed", "error_rmse", "error_med", "error_max_abs",
    ])
    for p in front_nmed:
        writer.writerow([
            p.fmt_name,
            p.is_fp,
            p.K,
            p.aca_window if p.aca_window is not None else "",
            f"{p.energy_pJ:.6f}",
            f"{p.energy_breakdown.multiplier_pJ:.6f}",
            f"{p.energy_breakdown.adder_pJ:.6f}",
            f"{p.error_nmed:.6e}",
            f"{p.error_rmse:.6e}",
            f"{p.error_med:.6e}",
            f"{p.error_max_abs:.6e}",
        ])
print(f"  Saved → {out_csv}  ({len(front_nmed)} Pareto-front points)")

# ============================================================
# 8.  Interactive window (skipped if no display)
# ============================================================

try:
    plt.show()
except Exception:
    pass  # headless environment — PNGs are the deliverable
