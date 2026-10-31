"""Isolated publication guards: synthetic figures only, no benchmark imports."""
import importlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_artifacts_bounds import (
    PlotBoundsError, register_bounds_data, register_bounds_image_data,
    validate_figure_bounds,
)


@pytest.fixture
def figure():
    fig, ax = plt.subplots()
    ax.set(xlim=(.1, 2), ylim=(.1, 2))
    yield fig, ax
    plt.close(fig)


@pytest.mark.parametrize("kind", ["line", "scatter", "interval", "errorbar"])
def test_clipped_observation_or_endpoint_fails(figure, kind):
    fig, ax = figure
    if kind == "line":
        ax.plot([1, 3], [1, 1], clip_on=False)
    elif kind == "scatter":
        ax.scatter([1, 3], [1, 1])
    elif kind == "interval":
        ax.hlines(1, .5, 3)
    else:
        ax.errorbar([1], [1], xerr=[[.5], [2]])
    with pytest.raises(PlotBoundsError, match="outside"):
        validate_figure_bounds(fig, "clipped")


@pytest.mark.parametrize("kind", ["line", "scatter", "interval"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_unmasked_nonfinite_never_disappears_silently(figure, kind, value):
    fig, ax = figure
    if kind == "line":
        ax.plot([1, value], [1, 1])
    elif kind == "scatter":
        ax.scatter([1, value], [1, 1])
    else:
        ax.add_collection(LineCollection([[(1, 1), (value, 1)]]))
    with pytest.raises(PlotBoundsError):
        validate_figure_bounds(fig)


@pytest.mark.parametrize("kind", ["line", "scatter", "interval"])
@pytest.mark.parametrize("value", [0, -1])
def test_log_domain_is_checked_before_transformation(figure, kind, value):
    fig, ax = figure
    ax.set_xscale("log")
    if kind == "line":
        ax.plot([value], [1])
    elif kind == "scatter":
        ax.scatter([value], [1])
    else:
        ax.hlines(1, value, 1)
    with pytest.raises(PlotBoundsError, match="nonpositive"):
        validate_figure_bounds(fig)


def test_decorations_and_colorbar_are_not_observations(figure):
    fig, ax = figure
    ax.plot([.5, 1.5], [1, 1])
    ax.axvline(100); ax.axhline(-100)
    ax.axvspan(20, 30); ax.axhspan(-30, -20)
    ax.text(500, 500, "outside annotation")
    ax.annotate("outside", xy=(400, 400), xytext=(500, 500))
    ax.legend(handles=[Line2D([100], [100], label="proxy")])
    image = ax.imshow([[1, 2], [3, 4]], extent=(.1, 2, .1, 2))
    fig.colorbar(image, ax=ax)
    coverage = validate_figure_bounds(fig)
    assert coverage == dict(axes=1, lines=1, collections=0, images=1, observations=6)


def test_reversed_log_and_symlog_axes_are_valid(figure):
    fig, ax = figure
    ax.set_xscale("log"); ax.set_xlim(2, .1)
    ax.set_yscale("symlog"); ax.set_ylim(2, -2)
    ax.plot([.1, 1, 2], [-2, 0, 2])
    assert validate_figure_bounds(fig)["observations"] == 3


def test_explicit_masked_line_gaps_are_retained(figure):
    fig, ax = figure
    ax.plot([.5, 1, 1.5], np.ma.array([1, np.nan, 1], mask=[0, 1, 0]))
    assert validate_figure_bounds(fig)["observations"] == 2


def test_scatter_mask_requires_original_inputs_and_stale_registration_fails(figure):
    fig, ax = figure
    x = np.ma.array([.5, np.nan, 1.5], mask=[0, 1, 0])
    y = np.array([1, 1, 1])
    points = ax.scatter(x, y)
    with pytest.raises(PlotBoundsError, match="ambiguous provenance"):
        validate_figure_bounds(fig)
    register_bounds_data(points, x, y)
    assert validate_figure_bounds(fig)["observations"] == 2
    points.set_offsets([[.5, 1], [1, 1], [1.5, 1]])
    with pytest.raises(PlotBoundsError, match="registered mask"):
        validate_figure_bounds(fig)


def test_registering_unmasked_nan_cannot_excuse_bad_scatter(figure):
    fig, ax = figure
    x, y = [.5, np.nan], [1, 1]
    register_bounds_data(ax.scatter(x, y), x, y)
    with pytest.raises(PlotBoundsError, match="nonfinite"):
        validate_figure_bounds(fig)


def test_masked_interval_gaps_require_original_inputs(figure):
    fig, ax = figure
    vertices = np.ma.array([[.5, 1], [np.nan, np.nan], [1.5, 1]],
                           mask=[[0, 0], [1, 1], [0, 0]])
    interval = LineCollection([vertices])
    ax.add_collection(interval)
    register_bounds_data(interval, vertices[:, 0], vertices[:, 1])
    assert validate_figure_bounds(fig)["observations"] == 2


@pytest.mark.parametrize("problem", ["extent", "nan", "logzero"])
def test_images_cannot_hide_invalid_values_or_cropped_cells(figure, problem):
    fig, ax = figure
    values = [[1, 2], [3, 4]]
    if problem == "nan": values[0][0] = np.nan
    if problem == "logzero": values[0][0] = 0
    ax.imshow(values, extent=(.1, 3 if problem == "extent" else 2, .1, 2),
              norm=LogNorm(.1, 4) if problem == "logzero" else None)
    with pytest.raises(PlotBoundsError):
        validate_figure_bounds(fig)


def test_explicitly_masked_image_is_valid(figure):
    fig, ax = figure
    values = np.ma.array([[1, np.nan], [3, 4]], mask=[[0, 1], [0, 0]])
    register_bounds_image_data(ax.imshow(values, extent=(.1, 2, .1, 2)), values)
    assert validate_figure_bounds(fig)["observations"] == 3


def test_translated_scatter_checks_displayed_data_coordinates(figure):
    fig, ax = figure
    points = ax.scatter([1], [1])
    points.set_offset_transform(Affine2D().translate(3, 0) + ax.transData)
    with pytest.raises(PlotBoundsError, match="outside"):
        validate_figure_bounds(fig)


def test_rotated_image_checks_all_four_corners(figure):
    fig, ax = figure
    image = ax.imshow([[1, 2], [3, 4]], extent=(.5, 1.5, .5, 1.5))
    image.set_transform(Affine2D().rotate_deg_around(1, 1, 45) + ax.transData)
    ax.set_xlim(.8, 1.2)
    # The transformed lower-left and upper-right corners are both at x=1;
    # the other two corners fall outside these limits.
    with pytest.raises(PlotBoundsError, match="outside"):
        validate_figure_bounds(fig)


def test_shifted_interval_collections_fail_closed(figure):
    fig, ax = figure
    interval = LineCollection([[(.5, 1), (1, 1)]], offsets=[(3, 0)])
    ax.add_collection(interval)
    with pytest.raises(PlotBoundsError, match="offset LineCollections"):
        validate_figure_bounds(fig)


@pytest.mark.parametrize("helper", ["common", "support"])
@pytest.mark.parametrize("kind", ["point", "interval"])
def test_save_helpers_fail_before_any_publication(figure, tmp_path, monkeypatch, helper, kind):
    fig, ax = figure
    if kind == "point": ax.plot([3], [1])
    else: ax.hlines(1, .5, 3)
    called = []
    monkeypatch.setattr(fig, "savefig", lambda *a, **k: called.append((a, k)))
    if helper == "common":
        module = importlib.import_module("paper_artifacts_common")
        monkeypatch.setattr(module, "DATA", tmp_path)
        monkeypatch.setattr(module, "FIGURES", tmp_path)
        before = dict(module.TYPOGRAPHY)
        save = lambda: module.save(fig, "invalid")
    else:
        module = importlib.import_module("paper_artifacts_support")
        save = lambda: module.save_figure(fig, tmp_path, "invalid")
    with pytest.raises(PlotBoundsError, match="outside"):
        save()
    assert called == []
    assert list(tmp_path.iterdir()) == []
    if helper == "common": assert module.TYPOGRAPHY == before
