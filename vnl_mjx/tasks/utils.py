"""Utility functions for scaling and recoloring geometries in a MuJoCo model, and rendering rollouts."""

import numpy as np
from typing import Any, Tuple

def _scale_vec(vec: list[float] | np.ndarray, s: float) -> None:
    """Scale a vector in-place by a scalar.

    Args:
        vec (list[float] | np.ndarray): The vector to scale.
        s (float): The scalar multiplier.

    Returns:
        None
    """
    for i in range(len(vec)):
        vec[i] *= s


def _scale_body_tree(body, s: float) -> None:
    """Recursively scale position, size, and fromto attributes on a body and its descendants.

    Args:
        body (Any): The body object to scale.
        s (float): The scalar multiplier.

    Returns:
        None
    """
    if hasattr(body, "pos"):
        _scale_vec(body.pos, s)

    for geom in body.geoms:
        if hasattr(geom, "pos"):
            _scale_vec(geom.pos, s)
        if hasattr(geom, "size"):
            _scale_vec(geom.size, s)
        if hasattr(geom, "fromto"):
            _scale_vec(geom.fromto, s)

    for site in body.sites:
        if hasattr(site, "pos"):
            _scale_vec(site.pos, s)
        if hasattr(site, "size"):
            _scale_vec(site.size, s)

    for joint in body.joints:
        if hasattr(joint, "pos"):
            _scale_vec(joint.pos, s)

    for child in body.bodies:
        _scale_body_tree(child, s)


def _recolour_geom(geom, rgba: list[float]) -> None:
    """Modify the color and collision group of a geometry, preserving original channels when rgba entry is -1."""
    # Capture original color channels
    original_rgba = list(geom.rgba)
    new_rgba = []
    # Build new RGBA, keeping original channel if new value is -1
    for orig_val, new_val in zip(original_rgba, rgba):
        if new_val == -1:
            new_rgba.append(orig_val)
        else:
            new_rgba.append(new_val)
    # If fewer new values provided, preserve any remaining original channels
    if len(original_rgba) > len(rgba):
        new_rgba.extend(original_rgba[len(rgba):])
    geom.rgba = new_rgba
    geom.group = 2  # separate collision group


def _recolour_tree(body, rgba: list[float]) -> None:
    """Recursively recolor all geometries in a body and its descendants."""
    for geom in body.geoms:
        _recolour_geom(geom, rgba)
    for child in body.bodies:
        _recolour_tree(child, rgba)
