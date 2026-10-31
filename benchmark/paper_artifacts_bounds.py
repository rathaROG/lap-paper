"""Fail before publication when observed figure data would be clipped.

This checks observation centres and interval endpoints, not marker glyphs or
text layout. It covers the current charts' data-coordinate lines, scatter
offsets, line collections and image extents. Mixed data/axes reference guides,
patches, annotations, legend proxies and colourbar axes are not observations.
Other data-coordinate collection types fail closed until explicitly supported.
LineCollection offsets also fail closed: the current interval plots store
their actual endpoints directly, without an additional offset transform.

An explicit mask is a legitimate missing observation; an unmasked NaN/Inf is
an error. Matplotlib discards that distinction for scatter and image inputs,
so their masked data require registering the original input with the helpers
below. Colour normalization may intentionally saturate; its domain (including
strict positivity for LogNorm), rather than its vmin/vmax, is checked.
"""
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import LogNorm


class PlotBoundsError(ValueError):
    """A plotted observation is invalid or outside the declared view."""


def register_bounds_data(artist, x, y):
    """Preserve explicitly masked raw scatter/interval coordinates.

Coordinates must describe the actual artist in the same order, flattening
interval endpoint arrays if needed. Registration is checked against the artist
at save time and cannot replace or excuse a changed, invalid observation.
"""
    artist._paper_bounds_data = (np.ma.array(x, copy=True), np.ma.array(y, copy=True))
    return artist


def register_bounds_image_data(image, values):
    """Preserve an image's explicitly masked original scalar/colour array."""
    image._paper_bounds_image_data = np.ma.array(values, copy=True)
    return image


def _fail(context, message):
    raise PlotBoundsError(f"{context}: {message}")


def _data_transform(transform, ax):
    # axhline/axvline blend axes fractions with data; neither is a full-data
    # transform. In particular, clip_on=False alone never excuses observations.
    return transform == ax.transData or transform.contains_branch(ax.transData)


def _coordinates(ax, x, y, context):
    try:
        x = np.ma.asarray(ax.convert_xunits(x), dtype=float).ravel()
        y = np.ma.asarray(ax.convert_yunits(y), dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        _fail(context, f"cannot interpret observation coordinates: {exc}")
    if x.shape != y.shape:
        _fail(context, f"different coordinate lengths: {len(x)} and {len(y)}")
    return np.ma.column_stack((x, y))


def _finite_values(values, context):
    if not np.isfinite(values).all():
        index = tuple(np.argwhere(~np.isfinite(values))[0])
        _fail(context, f"unmasked nonfinite value {values[index]!r} at {index}")


def _check_coordinates(ax, points, transform, context):
    points = np.ma.asarray(points, dtype=float).reshape(-1, 2)
    keep = ~np.ma.getmaskarray(points).any(axis=1)
    observed = np.asarray(points.data)[keep]
    _finite_values(observed, context)
    if transform != ax.transData and len(observed):
        observed = (transform - ax.transData).transform(observed)
        _finite_values(observed, context)
    for column, dimension in enumerate(("x", "y")):
        bounds = np.asarray(getattr(ax, f"get_{dimension}lim")(), dtype=float)
        _finite_values(bounds, context + f" {dimension} limits")
        lo, hi = sorted(bounds)
        values = observed[:, column]
        if getattr(ax, f"get_{dimension}scale")() == "log":
            if lo <= 0 or np.any(values <= 0):
                _fail(context, f"nonpositive {dimension} coordinate or limit on a log axis")
        # Permit only relative round-off at an endpoint, not a screen-size or
        # absolute tolerance that could hide small values on logarithmic axes.
        outside = ((values < lo) & ~np.isclose(values, lo, rtol=1e-12, atol=0)) | (
            (values > hi) & ~np.isclose(values, hi, rtol=1e-12, atol=0))
        if outside.any():
            value = values[np.flatnonzero(outside)[0]]
            _fail(context, f"{dimension}={value:.17g} outside [{lo:.17g}, {hi:.17g}]")
    return len(observed)


def _collection_points(artist, ax, points, context):
    points = np.ma.asarray(points, dtype=float).reshape(-1, 2)
    raw = getattr(artist, "_paper_bounds_data", None)
    if raw is None:
        if np.ma.getmaskarray(points).any():
            _fail(context, "masked collection coordinates have ambiguous provenance; "
                  "use register_bounds_data with explicitly masked original inputs")
        return points
    original = _coordinates(ax, *raw, context)
    if original.shape != points.shape:
        _fail(context, "registered coordinates no longer match the artist's shape")
    missing = np.ma.getmaskarray(original).any(axis=1)
    _finite_values(original.data[~missing], context)
    # Matplotlib may represent explicit missing collection vertices with NaNs
    # instead of a mask. Only positions masked in the original are exempt.
    actual_missing = np.ma.getmaskarray(points).any(axis=1) | ~np.isfinite(points.data).all(axis=1)
    if not np.array_equal(missing, actual_missing):
        _fail(context, "registered mask no longer matches the artist")
    if not np.array_equal(original.data[~missing], points.data[~missing]):
        _fail(context, "registered coordinates no longer match the artist")
    return original


def _check_image(image, ax, context):
    actual = np.ma.asarray(image.get_array())
    original = getattr(image, "_paper_bounds_image_data", None)
    if original is None:
        if np.ma.getmaskarray(actual).any():
            _fail(context, "masked image data have ambiguous provenance; "
                  "use register_bounds_image_data with the original input")
        values = actual
    else:
        values = np.ma.asarray(original)
        if values.shape != actual.shape or not np.array_equal(
                np.ma.getmaskarray(values), np.ma.getmaskarray(actual)):
            _fail(context, "registered image mask/shape no longer matches the artist")
        if not np.array_equal(values.compressed(), actual.compressed()):
            _fail(context, "registered image values no longer match the artist")
    observed = values.compressed()
    _finite_values(observed, context + " cells")
    if isinstance(image.norm, LogNorm) and np.any(observed <= 0):
        _fail(context, "nonpositive image cell under LogNorm")
    left, right, bottom, top = image.get_extent()
    _check_coordinates(ax, [[left, bottom], [left, top], [right, bottom], [right, top]],
                       image.get_transform(), context + " extent")
    return len(observed)


def validate_figure_bounds(fig, name="figure"):
    """Validate supported observations before drawing or writing any output.

Returns coverage counts for isolated audit/tests. It performs no file writes,
loads no saved benchmark data, and does not alter artists or axis limits.
"""
    coverage = dict(axes=0, lines=0, collections=0, images=0, observations=0)
    for index, ax in enumerate(fig.axes):
        if not ax.get_visible() or getattr(ax, "_colorbar", None) is not None:
            continue
        coverage["axes"] += 1
        base = f"{name}, axes {index + 1} ({ax.get_title()!r})"
        for i, line in enumerate(ax.lines):
            if not line.get_visible() or not _data_transform(line.get_transform(), ax):
                continue
            context = f"{base}, line {i + 1} ({line.get_label()!r})"
            points = _coordinates(ax, *line.get_data(orig=True), context)
            coverage["observations"] += _check_coordinates(ax, points, line.get_transform(), context)
            coverage["lines"] += 1
        for i, collection in enumerate(ax.collections):
            if not collection.get_visible():
                continue
            context = f"{base}, {type(collection).__name__} {i + 1}"
            if isinstance(collection, PathCollection):
                transform = collection.get_offset_transform()
                if not _data_transform(transform, ax):
                    continue
                points = collection.get_offsets()
            elif isinstance(collection, LineCollection):
                transform = collection.get_transform()
                if not _data_transform(transform, ax):
                    continue
                offsets = np.ma.asarray(collection.get_offsets())
                offset_transform = collection.get_offset_transform()
                if (np.ma.getmaskarray(offsets).any() or not np.isfinite(offsets.data).all()
                        or np.any(offsets.data != 0) or not offset_transform.is_affine
                        or not np.array_equal(offset_transform.get_matrix(), np.eye(3))):
                    _fail(context, "offset LineCollections require an explicit endpoint validator")
                paths = collection.get_paths()
                points = np.concatenate([path.vertices for path in paths]) if paths else np.empty((0, 2))
            else:
                if _data_transform(collection.get_transform(), ax):
                    _fail(context, "unsupported data-coordinate collection; add a bounds validator")
                continue
            points = _collection_points(collection, ax, points, context)
            coverage["observations"] += _check_coordinates(ax, points, transform, context)
            coverage["collections"] += 1
        for i, image in enumerate(ax.images):
            if image.get_visible() and _data_transform(image.get_transform(), ax):
                coverage["observations"] += _check_image(image, ax, f"{base}, image {i + 1}")
                coverage["images"] += 1
    return coverage
