"""Tests for visual grouping and publication style, without solving or rendering."""

import numpy as np
import pytest

from bo_t.data import load_frozen
from bo_t.plotting import (
    CASES,
    CASE_BY_NAME,
    OTHER_SELECTED,
    TRACK_COLORS,
    TRACK_MARKERS,
    legend_track_styles,
    render_case,
    visual_tracks,
)


def two_tracks(count=8):
    return np.asarray([
        (0.01 * (i + 1), real + 0.01 * i, -0.1 + 0.002 * i)
        for i in range(count) for real in (0.5, 1.5)
    ])


def test_cases_have_stable_sequential_names():
    assert [case.name for case in CASES] == [
        "es1d_beam", "umeda_ring_beam", "signed_em", "hh_ring_beam",
        "firehose", "muschietti_beam", "loss_cone_mirror",
    ]
    assert [case.figure_number for case in CASES] == list(range(1, 8))
    assert CASES[-1].slug == "fig07_loss_cone_mirror"
    assert sum(case.expected_scans for case in CASES) == 1061
    assert len(CASE_BY_NAME) == len(CASES)


def test_visual_tracks_are_deterministic_and_non_mutating():
    points = two_tracks()
    original = points.copy()
    ids, colors, audit = visual_tracks(points, CASE_BY_NAME["es1d_beam"])
    np.testing.assert_array_equal(ids, np.tile([0, 1], 8))
    np.testing.assert_array_equal(points, original)
    assert colors == {0: 0, 1: 1}
    assert audit["track_lengths"] == {"0": 8, "1": 8}
    assert audit["colored_point_count"] == 16
    assert audit["muted_short_track_point_count"] == 0
    assert audit["uses_reference_branches"] is False
    assert audit["colors_are_physical_mode_identities"] is False
    again = visual_tracks(points, CASE_BY_NAME["es1d_beam"])
    np.testing.assert_array_equal(again[0], ids)
    assert again[1:] == (colors, audit)


def test_visual_grouping_does_not_depend_on_input_row_order():
    points = two_tracks()
    order = np.random.default_rng(184).permutation(len(points))
    reference_ids, reference_colors, reference_audit = visual_tracks(
        points, CASE_BY_NAME["es1d_beam"])
    ids, colors, audit = visual_tracks(points[order], CASE_BY_NAME["es1d_beam"])
    np.testing.assert_array_equal(ids[np.argsort(order)], reference_ids)
    assert colors == reference_colors
    assert audit == reference_audit


def test_missing_scan_terminates_visual_segments():
    points = two_tracks(9)
    points = points[points[:, 0] != 0.05]
    ids, colors, audit = visual_tracks(points, CASE_BY_NAME["es1d_beam"])
    assert set(ids[points[:, 0] < 0.05]).isdisjoint(
        set(ids[points[:, 0] > 0.05]))
    assert audit["track_count"] == 4
    assert colors == {}
    assert audit["muted_short_track_point_count"] == len(points)


def test_independent_signed_halves_are_never_joined():
    x = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02])
    points = np.column_stack((x, 1.0 + x, np.full(len(x), -0.1)))
    ids, _, _ = visual_tracks(points, CASE_BY_NAME["signed_em"])
    assert ids[0] == ids[1]
    assert ids[1] != ids[2]
    assert ids[2] == ids[3] == ids[4]


def test_ambiguous_matches_do_not_connect_candidates():
    points = np.asarray([
        (0.01, -0.1, 0.0), (0.01, 0.1, 0.0),
        (0.02, 0.0, 0.0), (0.02, 0.0, 0.0),
    ])
    ids, colors, audit = visual_tracks(points, CASE_BY_NAME["es1d_beam"])
    assert len(set(ids)) == 4
    assert colors == {}
    assert audit["track_count"] == 4


def test_empty_tracks_are_supported():
    ids, colors, audit = visual_tracks(np.empty((0, 3)), CASES[0])
    assert ids.shape == (0,)
    assert colors == {}
    assert audit["track_count"] == 0


def test_legend_represents_all_and_only_used_styles():
    styles = legend_track_styles({7: 4, 8: 0, 9: 4}, include_muted=True)
    assert styles == (
        (TRACK_COLORS[0], TRACK_MARKERS[0], False),
        (TRACK_COLORS[4], TRACK_MARKERS[4], False),
        (OTHER_SELECTED, "o", True),
    )
    assert legend_track_styles({}, include_muted=True) == (
        (OTHER_SELECTED, "o", True),
    )
    assert legend_track_styles({}, include_muted=False) == ()


def test_unknown_case_fails_before_creating_output(tmp_path):
    target = tmp_path / "not-created"
    with pytest.raises(ValueError, match="Unknown case"):
        render_case("not-a-case", two_tracks(), two_tracks(), target)
    assert not target.exists()


@pytest.mark.parametrize("bad", [np.empty((0,)), np.empty((2, 4)), [[0, 1]]])
def test_wrong_shape_is_rejected_before_rendering(tmp_path, bad):
    with pytest.raises(ValueError, match="Invalid"):
        render_case("es1d_beam", bad, two_tracks(), tmp_path)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_data_are_rejected_before_rendering(tmp_path, value):
    points = two_tracks()
    points[0, 1] = value
    with pytest.raises(ValueError, match="Non-finite"):
        render_case("es1d_beam", points, two_tracks(), tmp_path)


def test_no_selected_points_does_not_silently_draw_raw_only(tmp_path):
    with pytest.raises(ValueError, match="No plotted data"):
        render_case("es1d_beam", two_tracks(), np.empty((0, 3)), tmp_path)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_frozen_visual_tracks_and_legend_are_consistent(case):
    """Audit complete published candidates without creating figure files."""
    _, selected, _ = load_frozen(case.name)

    def visible(growth_window):
        return (
            (case.xlim[0] <= selected[:, 0])
            & (selected[:, 0] <= case.xlim[1])
            & (case.real_window[0] <= selected[:, 1])
            & (selected[:, 1] <= case.real_window[1])
            & (growth_window[0] <= selected[:, 2])
            & (selected[:, 2] <= growth_window[1])
        )

    displayed = selected[
        visible(case.real_panel_growth_window or case.growth_window)
        | visible(case.growth_window)
    ]
    assert len(displayed) > 0
    ids, colors, audit = visual_tracks(displayed, case)
    for scan in np.unique(displayed[:, 0]):
        scan_ids = ids[displayed[:, 0] == scan]
        assert len(set(scan_ids)) == len(scan_ids)
    if case.name == "signed_em":
        assert set(ids[displayed[:, 0] < 0]).isdisjoint(
            set(ids[displayed[:, 0] >= 0]))

    order = np.random.default_rng(184).permutation(len(displayed))
    shuffled_ids, shuffled_colors, shuffled_audit = visual_tracks(displayed[order], case)
    # Compare multisets so exactly coincident candidates remain interchangeable.
    canonical = sorted((*point, int(track_id)) for point, track_id in zip(displayed, ids))
    shuffled = sorted((*point, int(track_id))
                      for point, track_id in zip(displayed[order], shuffled_ids))
    assert shuffled == canonical
    assert shuffled_colors == colors
    assert shuffled_audit == audit

    actual_styles = {
        (TRACK_COLORS[colors[int(track_id)]], TRACK_MARKERS[colors[int(track_id)]], False)
        if int(track_id) in colors else (OTHER_SELECTED, "o", True)
        for track_id in ids
    }
    legend = legend_track_styles(colors, audit["muted_short_track_point_count"] > 0)
    assert set(legend) == actual_styles
    assert len(legend) == len(actual_styles)
    assert audit["uses_reference_branches"] is False
    assert audit["colors_are_physical_mode_identities"] is False
