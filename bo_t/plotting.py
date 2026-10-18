"""Portable publication plots for the seven BO-T numerical examples.

Inputs are numerical arrays with columns [scan coordinate, real frequency,
growth rate]. Raw means the dynamic BO spectrum after declared auxiliary
poles and zero/constraint roots have been excluded. Colours distinguish visual
tracks only; they are not independently validated physical-mode identities.
No reference branch, archived receipt, or numerical solver is loaded here.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


WIDTH_IN, HEIGHT_IN = 7.15, 3.60
GRAY = "#80868B"
OTHER_SELECTED = "#A6C8D8"
TRACK_COLORS = ("#0072B2", "#009E73", "#56B4E9",
                "#E69F00", "#CC79A7", "#D55E00")
TRACK_MARKERS = ("o", "s", "^", "D", "v", "P")
MIN_COLORED_TRACK_POINTS = 5


def legend_track_styles(color_indices: dict[int, int],
                        include_muted: bool) -> tuple[tuple[str, str, bool], ...]:
    """Return only the marker styles actually used by displayed candidates."""
    styles = [(TRACK_COLORS[index], TRACK_MARKERS[index], False)
              for index in sorted(set(color_indices.values()))]
    if include_muted:
        styles.append((OTHER_SELECTED, "o", True))
    return tuple(styles)


@dataclass(frozen=True)
class Case:
    name: str
    figure_number: int
    x_label: str
    norm_label: str
    xlim: tuple[float, float]
    real_window: tuple[float, float]
    growth_window: tuple[float, float]
    xticks: tuple[float, ...]
    expected_scans: int
    real_panel_growth_window: tuple[float, float] | None = None

    @property
    def slug(self) -> str:
        return f"fig{self.figure_number:02d}_{self.name}"


CASES = (
    Case("es1d_beam", 1, r"$k\lambda_{De}$", r"\omega_{pe}",
         (0.0, 0.51), (-1.5, 3.05), (-0.5, 0.3),
         (0.0, 0.1, 0.2, 0.3, 0.4, 0.5), 50),
    Case("umeda_ring_beam", 2, r"$k\lambda_{De}$", r"\Omega_{ce}",
         (0.0, 0.305), (0.0, 8.0), (-2.0, 0.5),
         (0.0, 0.1, 0.2, 0.3), 117),
    Case("signed_em", 3, r"$k_zc/\Omega_{ce}$", r"\Omega_{ce}",
         (-2.56, 2.56), (-0.04, 3.04), (-0.72, 0.04),
         (-2.0, -1.0, 0.0, 1.0, 2.0), 500),
    Case("hh_ring_beam", 4, r"$k\lambda_{De}$", r"\Omega_{ce}",
         (0.0, 0.305), (0.0, 10.0), (-2.0, 0.5),
         (0.0, 0.1, 0.2, 0.3), 117),
    Case("firehose", 5, r"$kv_A/\Omega_{ci}$", r"\Omega_{ci}",
         (0.045, 0.505), (-0.25, 0.25), (-0.225, 0.09),
         (0.1, 0.2, 0.3, 0.4, 0.5), 46),
    Case("muschietti_beam", 6, r"$kc/\omega_{pi}$", r"\Omega_{ci}",
         (8.0, 77.0), (0.0, 600.0), (-5.0, 15.0),
         (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0), 131),
    Case("loss_cone_mirror", 7, r"$k_z/k_n$", r"\omega_n",
         (0.0, 1.055), (-2.52, 1.02), (-0.25, 0.08),
         (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), 100,
         real_panel_growth_window=(-1.02, 0.105)),
)
CASE_BY_NAME = {case.name: case for case in CASES}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visual_tracks(points: np.ndarray, case: Case) -> tuple[np.ndarray, dict[int, int], dict]:
    """Conservatively group plotted candidates using only adjacent scan roots.

    These IDs control marker appearance; they are neither solver output nor
    independently validated physical-mode identities. Ambiguous matches and
    missing scan points terminate a segment. No paper/archive data are used.
    """
    if len(points) == 0:
        return np.empty(0, dtype=int), {}, {"track_count": 0}
    x_values = np.unique(points[:, 0])
    nominal_step = float(np.median(np.diff(x_values))) if len(x_values) > 1 else 0.0
    groups = []
    for x in x_values:
        indices = np.flatnonzero(points[:, 0] == x)
        indices = indices[np.lexsort((points[indices, 2], points[indices, 1]))]
        groups.append(indices)

    def adjacent(left: np.ndarray, right: np.ndarray) -> bool:
        if nominal_step <= 0 or len(left) == 0 or len(right) == 0:
            return False
        x_left, x_right = points[left[0], 0], points[right[0], 0]
        if x_right - x_left > 1.5 * nominal_step + 1e-10:
            return False
        # The two signed-EM halves were frozen in independent runs.
        if case.slug == "fig03_signed_em" and x_left < 0 <= x_right:
            return False
        return True

    def roots(indices: np.ndarray) -> np.ndarray:
        return points[indices, 1] + 1j * points[indices, 2]

    calibration = []
    for previous, current in zip(groups, groups[1:]):
        if not adjacent(previous, current):
            continue
        distances = abs(roots(previous)[:, None] - roots(current)[None, :])
        nearest_current = np.argmin(distances, axis=1)
        nearest_previous = np.argmin(distances, axis=0)
        calibration.extend(
            float(distances[i, j]) for i, j in enumerate(nearest_current)
            if nearest_previous[j] == i
        )
    gate = 1.6 * float(np.quantile(calibration, 0.90)) if calibration else 0.0

    track_ids = np.full(len(points), -1, dtype=int)
    histories: dict[int, list[int]] = {}
    next_track_id = 0
    for layer, current in enumerate(groups):
        matches: dict[int, int] = {}
        if layer and adjacent(groups[layer - 1], current) and gate > 0:
            previous = groups[layer - 1]
            previous_roots = roots(previous)
            current_roots = roots(current)
            predicted = previous_roots.copy()
            for i, point_index in enumerate(previous):
                history = histories[int(track_ids[point_index])]
                if len(history) < 2:
                    continue
                before = history[-2]
                dx = points[point_index, 0] - points[before, 0]
                if dx > 0:
                    predicted[i] += ((previous_roots[i]
                                      - complex(points[before, 1], points[before, 2]))
                                     * (points[current[0], 0] - points[point_index, 0]) / dx)
            predicted_distance = abs(predicted[:, None] - current_roots[None, :])
            actual_distance = abs(previous_roots[:, None] - current_roots[None, :])
            nearest_current = np.argmin(predicted_distance, axis=1)
            nearest_previous = np.argmin(predicted_distance, axis=0)
            for i, j in enumerate(nearest_current):
                if nearest_previous[j] != i:
                    continue
                best = float(predicted_distance[i, j])
                if best > gate or actual_distance[i, j] > gate:
                    continue
                row_second = (float(np.partition(predicted_distance[i], 1)[1])
                              if len(current) > 1 else float("inf"))
                col_second = (float(np.partition(predicted_distance[:, j], 1)[1])
                              if len(previous) > 1 else float("inf"))
                if (row_second - best < max(0.25 * best, 1e-10)
                        or col_second - best < max(0.25 * best, 1e-10)):
                    continue
                matches[int(current[j])] = int(track_ids[previous[i]])
        for point_index in current:
            point_index = int(point_index)
            if point_index in matches:
                track_id = matches[point_index]
            else:
                track_id = next_track_id
                histories[track_id] = []
                next_track_id += 1
            track_ids[point_index] = track_id
            histories[track_id].append(point_index)
    if np.any(track_ids < 0):
        raise AssertionError("Every displayed BO-T candidate needs a visual track")
    for group in groups:
        if len(set(track_ids[group])) != len(group):
            raise AssertionError("A visual track cannot contain two roots at one scan point")

    long_ids = [track_id for track_id, history in histories.items()
                if len(history) >= MIN_COLORED_TRACK_POINTS]
    ordered = sorted(long_ids, key=lambda t: (-len(histories[t]),
                                              points[histories[t][0], 0],
                                              points[histories[t][0], 1],
                                              points[histories[t][0], 2]))
    # Reuse a color only when the already-colored trajectories are well
    # separated in the complex plane at their shared scan points.
    real_span = case.real_window[1] - case.real_window[0]
    growth_span = (case.real_panel_growth_window or case.growth_window)[1] - (
        case.real_panel_growth_window or case.growth_window)[0]
    positions = {
        track_id: {float(points[i, 0]): (float(points[i, 1]), float(points[i, 2]))
                   for i in histories[track_id]}
        for track_id in ordered
    }
    color_indices: dict[int, int] = {}
    usage = [0] * len(TRACK_COLORS)
    for track_id in ordered:
        penalty = [0.0] * len(TRACK_COLORS)
        for existing_id, color_index in color_indices.items():
            common = positions[track_id].keys() & positions[existing_id].keys()
            if not common:
                continue
            minimum = min(
                np.hypot((positions[track_id][x][0] - positions[existing_id][x][0])
                         / real_span,
                         (positions[track_id][x][1] - positions[existing_id][x][1])
                         / growth_span)
                for x in common
            )
            penalty[color_index] += 1.0 / (0.04 + minimum)
        chosen = min(range(len(TRACK_COLORS)),
                     key=lambda color: (penalty[color], usage[color], color))
        color_indices[track_id] = chosen
        usage[chosen] += 1
    metadata = {
        "method": "adjacent_scan_secant_mutual_nearest_visual_grouping",
        "uses_reference_branches": False,
        "colors_are_physical_mode_identities": False,
        "nominal_scan_step": nominal_step,
        "maximum_match_distance": gate,
        "track_count": len(histories),
        "colored_track_count": len(long_ids),
        "colored_point_count": sum(len(histories[t]) for t in long_ids),
        "muted_short_track_point_count": sum(len(histories[t]) for t in histories
                                            if t not in color_indices),
        "track_lengths": {str(t): len(history) for t, history in histories.items()},
        "track_color_indices": {str(t): color_indices[t] for t in ordered},
    }
    return track_ids, color_indices, metadata


def render(case: Case, raw: np.ndarray, selected: np.ndarray,
           output_dir: Path) -> dict:
    """Render supplied arrays without changing numerical selections."""
    raw = np.asarray(raw, dtype=float)
    selected = np.asarray(selected, dtype=float)
    if raw.ndim != 2 or selected.ndim != 2 or raw.shape[1] != 3 or selected.shape[1] != 3:
        raise ValueError(f"Invalid [k, real, growth] arrays for {case.slug}")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(selected)):
        raise ValueError(f"Non-finite [k, real, growth] arrays for {case.name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xlo, xhi = case.xlim
    rlo, rhi = case.real_window

    # The real-frequency range is shared between panels: a high-frequency
    # root cannot create a spurious gamma ~= 0 band by projection.  The mirror
    # case gives gamma its own tighter vertical window to resolve its peaks.
    def visible(points: np.ndarray, growth_window: tuple[float, float]) -> np.ndarray:
        vlo, vhi = growth_window
        return ((xlo <= points[:, 0]) & (points[:, 0] <= xhi)
                & (rlo <= points[:, 1]) & (points[:, 1] <= rhi)
                & (vlo <= points[:, 2]) & (points[:, 2] <= vhi))

    real_growth_window = case.real_panel_growth_window or case.growth_window
    source_real_mask = visible(raw, real_growth_window)
    source_growth_mask = visible(raw, case.growth_window)
    selected_real_mask = visible(selected, real_growth_window)
    selected_growth_mask = visible(selected, case.growth_window)
    source_real = raw[source_real_mask]
    source_growth = raw[source_growth_mask]
    selected_real = selected[selected_real_mask]
    selected_growth = selected[selected_growth_mask]
    if min(map(len, (source_real, selected_real, source_growth, selected_growth))) == 0:
        raise ValueError(f"No plotted data for {case.slug}")
    display_mask = selected_real_mask | selected_growth_mask
    display_indices = np.flatnonzero(display_mask)
    visual_ids, color_indices, track_metadata = visual_tracks(selected[display_mask], case)
    all_visual_ids = np.full(len(selected), -1, dtype=int)
    all_visual_ids[display_mask] = visual_ids
    real_ids = all_visual_ids[selected_real_mask]
    growth_ids = all_visual_ids[selected_growth_mask]

    # Density-adaptive areas preserve the reference figure's open colored
    # markers without turning the largest scans into thick ribbons.
    raw_size = 5.5 if len(source_real) < 5000 else 2.5
    raw_alpha = 0.68 if len(source_real) < 5000 else 0.52
    # Marker width follows horizontal scan spacing. A dense 500-point scan
    # needs much smaller marks than the 46- or 50-point examples.
    if case.expected_scans >= 400:
        selected_size, selected_edge_width, muted_size = 1.8, 0.34, 1.4
    elif case.expected_scans >= 110:
        selected_size, selected_edge_width, muted_size = 4.5, 0.50, 3.0
    elif case.expected_scans >= 90:
        selected_size, selected_edge_width, muted_size = 7.5, 0.65, 4.0
    else:
        selected_size, selected_edge_width, muted_size = 16.0, 0.80, 7.0

    with plt.rc_context({
        "font.family": "DejaVu Serif",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, HEIGHT_IN))
        fig.subplots_adjust(left=0.085, right=0.982, bottom=0.27, top=0.975,
                            wspace=0.29)
        for axis, ordinate, window, ylabel, source_view, selected_view, selected_ids in (
            (axes[0], 1, case.real_window, rf"$\omega_r/{case.norm_label}$",
             source_real, selected_real, real_ids),
            (axes[1], 2, case.growth_window, rf"$\gamma/{case.norm_label}$",
             source_growth, selected_growth, growth_ids),
        ):
            axis.scatter(source_view[:, 0], source_view[:, ordinate],
                         s=raw_size, marker=".", c=GRAY, alpha=raw_alpha,
                         linewidths=0, rasterized=False, zorder=1)
            muted_mask = np.fromiter((int(t) not in color_indices for t in selected_ids),
                                     dtype=bool, count=len(selected_ids))
            if np.any(muted_mask):
                axis.scatter(selected_view[muted_mask, 0],
                             selected_view[muted_mask, ordinate],
                             s=muted_size, marker="o", color=OTHER_SELECTED,
                             alpha=0.70, linewidths=0, rasterized=False, zorder=2)
            for color_index, (color, marker) in enumerate(zip(TRACK_COLORS,
                                                               TRACK_MARKERS, strict=True)):
                mask = np.fromiter((color_indices.get(int(t), -1) == color_index
                                    for t in selected_ids),
                                   dtype=bool, count=len(selected_ids))
                if not np.any(mask):
                    continue
                axis.scatter(selected_view[mask, 0], selected_view[mask, ordinate],
                             s=selected_size, marker=marker, facecolors="none",
                             edgecolors=color, linewidths=selected_edge_width,
                             alpha=0.95, rasterized=False, zorder=3)
            axis.set_xlim(*case.xlim)
            axis.set_ylim(*window)
            axis.set_xticks(case.xticks)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.set_xlabel(case.x_label)
            axis.set_ylabel(ylabel)
            axis.set_axisbelow(True)
            axis.grid(True, color="#E4E4E4", linewidth=0.5)
            axis.tick_params(direction="out", length=3.0, width=0.7)
        axes[1].axhline(0, color="#777777", lw=0.55, zorder=0)
        spectrum_handle = Line2D([], [], ls="", marker="o", markersize=4.5,
                                 markerfacecolor=GRAY, markeredgecolor="none")
        selected_handle = tuple(
            Line2D([], [], ls="", marker=marker, markersize=5.2,
                   markerfacecolor=color if filled else "none",
                   markeredgecolor="none" if filled else color,
                   markeredgewidth=0.9)
            for color, marker, filled in legend_track_styles(
                color_indices, track_metadata["muted_short_track_point_count"] > 0)
        )
        fig.legend(handles=(spectrum_handle, selected_handle),
                   labels=("BO spectrum", "BO-T"),
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.05)},
                   loc="lower center", bbox_to_anchor=(0.5, 0.025),
                   ncol=2, frameon=False, fontsize=8.5, columnspacing=1.8,
                   handlelength=6.2, handletextpad=0.55)
        stem = output_dir / case.slug
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".png"), dpi=260)
        plt.close(fig)

    track_path = output_dir / f"{case.slug}_visual_tracks.csv"
    with track_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("selected_row_0based", "scan_x", "root_real", "root_imag",
                         "visual_track_id", "track_length", "color", "marker"))
        for local_index, original_index in enumerate(display_indices):
            track_id = int(visual_ids[local_index])
            color_index = color_indices.get(track_id)
            writer.writerow((int(original_index), *selected[original_index], track_id,
                             track_metadata["track_lengths"][str(track_id)],
                             TRACK_COLORS[color_index] if color_index is not None
                             else OTHER_SELECTED,
                             TRACK_MARKERS[color_index] if color_index is not None else "o"))
    return {
        "case_name": case.name,
        "files": {
            **{extension: {"name": stem.with_suffix("." + extension).name,
                           "sha256": digest(stem.with_suffix("." + extension))}
               for extension in ("pdf", "svg", "png")},
            "visual_tracks": {"name": track_path.name, "sha256": digest(track_path)},
        },
        "source_dynamic_root_count": len(raw),
        "selected_root_count": len(selected),
        "plotted_source_root_count": {"real": len(source_real), "growth": len(source_growth)},
        "plotted_selected_root_count": {"real": len(selected_real), "growth": len(selected_growth)},
        "shared_real_frequency_window": list(case.real_window),
        "growth_window": list(case.growth_window),
        "real_panel_growth_gate": list(real_growth_window),
        "xlim": list(case.xlim),
        "visual_tracking": track_metadata,
        "visual_track_csv_sha256": digest(track_path),
        "pdf_sha256": digest(stem.with_suffix(".pdf")),
        "svg_sha256": digest(stem.with_suffix(".svg")),
        "png_sha256": digest(stem.with_suffix(".png")),
    }



def render_case(case_name: str, raw: np.ndarray, selected: np.ndarray,
                output_dir: Path) -> dict:
    """Write a case's PDF, SVG, PNG and visual-track CSV to output_dir.

    Returned filenames are relative to output_dir. The audit contains counts
    before and after viewport clipping, tracking diagnostics and file hashes.
    This entry point accepts partial scans so smoke runs can use the identical
    viewport and style; scan completeness must be recorded by the caller.
    """
    try:
        case = CASE_BY_NAME[case_name]
    except KeyError:
        raise ValueError(f"Unknown case {case_name!r}; choose from "
                         f"{', '.join(CASE_BY_NAME)}") from None
    return render(case, raw, selected, output_dir)

