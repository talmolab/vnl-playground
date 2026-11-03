"""Y-maze and T-maze geometry for the vision-guided spout-maze task.

``build_y_maze`` / ``build_t_maze`` run once, host-side, in an environment's
``__init__`` and are PURE NUMPY.  Each returns the wall boxes (to be added as
MuJoCo box geoms) plus the key points of the maze, as one layout type
(`YMazeLayout`, alias `MazeLayout`).  ``locate`` and ``segment_of`` are the only
functions meant to run under ``jax.jit``: ``locate`` maps a world xy position
to ``(arm_id, distance_to_arm_end)`` for the trial state machine, and
``segment_of`` maps it to a T-maze segment label for offline analysis.

Layout (top view, +y up, the rat starts at the bottom facing +y)::

        arm 0 (-x)  \\  /  arm 1 (+x)        arms at +-60 deg from +y,
                     \\/                      i.e. 120 deg apart
                      J  (junction, origin)
                      |
                      |  stem (along -y)
                      |
                      S  (start; the stem's back wall is half a corridor
                          width behind it, so the rat's tail has room)

Every corridor extends half a corridor width past its nominal end point (the
start point ``S`` and each ``arm_end_xy``), where the end cap's inner face is.

With the default 120 deg arm angle the stem and both arms are mutually 120 deg
apart, so the junction is an equilateral triangle of side ``corridor_width``
whose three sides are the corridor mouths.

The free space is the union of three constant-width corridors.  Its outline is a
9-edge polygon (stem back cap, stem right wall, right-arm lower wall, right-arm
cap, right-arm upper wall, left-arm upper wall, left-arm cap, left-arm lower
wall, stem left wall), traversed counter-clockwise.  Each edge becomes one thin
box whose *inner face* lies exactly on the polygon edge, so the clear width of
every corridor is ``corridor_width`` (not reduced by the wall thickness).  The
three end caps are widened to cover the outer corners.

Conventions
-----------
* Box geoms: ``wall_pos[i]`` is the box centre (z = height / 2),
  ``wall_halfsize[i] = (half-length along the wall, half-thickness,
  half-height)``, ``wall_yaw[i]`` rotates the box about +z (local x along the
  wall).
* ``start_yaw`` follows the rodent root convention (yaw 0 faces +x), so
  ``pi / 2`` faces +y, i.e. the junction.
* Arm ``0`` is the ``-x`` (left, as seen by a rat facing +y) arm and arm ``1``
  is the ``+x`` (right) arm.  That is ``+60`` and ``-60`` deg from +y with
  counter-clockwise-positive angles.

T maze (`build_t_maze`)
-----------------------
The lab's mouse T maze (``data/tmaze_spec.png``: corridors 10 cm, a stem of
five 20 cm segments, a 10 x 10 cm junction, two 27.5 cm arms) built at x2.5 for
the virtual rat.  All dimensions are CLEAR inner dimensions::

        arm_left (arm 0, -x)    J    arm_right (arm 1, +x)     arm row:
      |------- 0.6875 ------|0.25|------- 0.6875 -------|       y in [-0.125, 0.125]
                             seg4
                             seg3    stem along -y: five 0.50 segments,
                             seg2    x in [-0.125, 0.125]
                             seg1
                             start   back-wall inner face at y = -2.625

* The junction square (0.25 x 0.25) is centred on the origin; the inner
  footprint is 1.625 x 2.75.
* The clear outline is an 8-edge polygon (stem back cap, stem right wall,
  right-arm lower wall, right end cap, the top wall spanning both arms, left
  end cap, left-arm lower wall, stem left wall), with walls extruded outward
  exactly as for the Y (the three caps widened by one wall thickness to close
  the outer corners).
* The start pose is the centre of the ``start`` segment, facing +y.  Each arm's
  spout point (``arm_end_xy`` = ``treat_xy``) is on its centreline,
  ``spout_offset`` in front of the end wall's inner face.
* ``arm_mouth`` is the arm's mouth (|x| = corridor_width / 2) PLUS
  ``entry_margin``, so `locate` reports an arm only once the torso is that far
  into it.  ``locate`` itself is shape-agnostic (it only reads ``arm_dir``,
  ``arm_end_xy``, ``junction_xy`` and ``arm_mouth``); in the T, every
  free-space point past the mouth line lies in the arm row.
* `segment_of` labels a position with the T maze's segments (`T_SEGMENT_NAMES`).
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

__all__ = [
    "T_SEGMENT_NAMES",
    "MazeLayout",
    "TMazeSpec",
    "TSegments",
    "YMazeLayout",
    "YMazeSpec",
    "build_t_maze",
    "build_y_maze",
    "locate",
    "segment_of",
    "t_segment_names",
]


@dataclasses.dataclass(frozen=True)
class YMazeSpec:
    """Dimensions of the Y-maze, in metres / degrees.

    Attributes:
      stem_length: junction -> start point, along -y.  The stem's back wall is
        ``corridor_width / 2`` behind the start point.
      arm_length: junction -> arm end point, measured along the arm axis.  The
        arm's end cap is ``corridor_width / 2`` further out.
      arm_angle_deg: angle between the two arms (each is at half of this from
        +y).
      corridor_width: clear width between the facing inner wall faces.  0.465 m
        is the measured rat turn-around width (the rat is ~0.31 x 0.08 m).
      wall_thickness: box thickness; walls extend *outward* from the corridor.
      wall_height: full box height.
      appear_distance: the spout "appears" when the rat is within this
        distance of the arm end (last 0.5 m of an arm).  Used by the env.
      spout_radius: radius of the spout marker.  Used by the env.
      treat_inset: the treat sits this far from the arm end, back along the
        arm.
    """

    stem_length: float = 0.9
    arm_length: float = 0.9
    arm_angle_deg: float = 120.0
    corridor_width: float = 0.465
    wall_thickness: float = 0.15
    wall_height: float = 0.3
    appear_distance: float = 0.5
    spout_radius: float = 0.15
    treat_inset: float = 0.15


@dataclasses.dataclass(frozen=True)
class TMazeSpec:
    """Dimensions of the T maze (the lab's mouse T maze at x2.5), in metres.

    All lengths are CLEAR inner dimensions (between wall inner faces).

    Attributes:
      corridor_width: clear width of the stem, the junction and both arms.
      segment_length: length of each stem segment along y.
      n_stem_segments: number of stem segments (``start``, ``seg1`` ...); the
        stem is ``n_stem_segments * segment_length`` long, from its back wall to
        the junction edge.
      arm_length: from the junction edge to the arm end wall's inner face.
      wall_thickness: box thickness; walls extend *outward* from the outline.
      wall_height: full box height.
      spout_offset: the spout / treat point sits this far in front of the arm
        end wall's inner face, on the arm's centreline.
      entry_margin: `locate` reports an arm only once the position is this far
        past the arm's mouth (|x| = corridor_width / 2).
    """

    corridor_width: float = 0.25
    segment_length: float = 0.50
    n_stem_segments: int = 5
    arm_length: float = 0.6875
    wall_thickness: float = 0.05
    wall_height: float = 0.30
    spout_offset: float = 0.10
    entry_margin: float = 0.03


class TSegments(NamedTuple):
    """Axis-aligned segment bounds of a T maze (world frame, junction at the origin).

    Attributes:
      stem_edges_y: ``(n + 1,)`` increasing y of the stem segment boundaries,
        from the back wall's inner face to the junction edge (``-half_width``).
        Segment ``k`` covers ``[stem_edges_y[k], stem_edges_y[k + 1])``.
      half_width: half the clear corridor width.
      arm_reach: ``|x|`` of the arm end walls' inner faces.
    """

    stem_edges_y: np.ndarray
    half_width: float
    arm_reach: float


def t_segment_names(n_stem_segments: int = 5) -> tuple[str, ...]:
    """Segment names in `segment_of` id order for a T with ``n_stem_segments``."""
    stem = ("start", *(f"seg{k}" for k in range(1, n_stem_segments)))
    return (*stem, "junction", "arm_left", "arm_right")


#: `segment_of` ids of the default T maze: 0 = ``start`` ... 4 = ``seg4``,
#: 5 = ``junction``, 6 = ``arm_left``, 7 = ``arm_right`` (-1 = outside).
T_SEGMENT_NAMES = t_segment_names(5)


class YMazeLayout(NamedTuple):
    """Wall boxes and key points of a Y or T maze (numpy float64, world frame).

    The T maze (`build_t_maze`) returns this same type; `MazeLayout` is a
    neutral alias.

    Attributes:
      wall_pos: ``(N, 3)`` box centres.
      wall_halfsize: ``(N, 3)`` half-sizes (x along the wall, y thickness, z height).
      wall_yaw: ``(N,)`` rotation of each box about +z.
      junction_xy: ``(2,)`` the junction (origin).
      start_xy: ``(2,)`` the rat's start point, at the far end of the stem.
      start_yaw: facing of the start pose (towards the junction).
      arm_dir: ``(2, 2)`` unit vectors from the junction along each arm.
      arm_end_xy: ``(2, 2)`` arm end points on each centreline: Y, half a
        corridor width in front of the end cap; T, the spout point
        (``spout_offset`` in front of the end wall).
      treat_xy: ``(2, 2)`` treat positions: Y, ``treat_inset`` before the arm
        end; T, equal to ``arm_end_xy``.
      extent: half-width of the origin-centred bounding square of all wall
        boxes (max over box corners of ``max(|x|, |y|)``).
      arm_mouth: axial distance from the junction, along an arm's axis, beyond
        which ``locate`` reports that arm.  Y: where the arm's corridor begins
        (the mouth line).  T: the mouth (|x| = corridor_width / 2) plus
        ``entry_margin``.
      segments: T only, the `segment_of` table (`TSegments`); ``None`` for a Y
        layout, for which `segment_of` returns -1 everywhere.
    """

    wall_pos: np.ndarray
    wall_halfsize: np.ndarray
    wall_yaw: np.ndarray
    junction_xy: np.ndarray
    start_xy: np.ndarray
    start_yaw: float
    arm_dir: np.ndarray
    arm_end_xy: np.ndarray
    treat_xy: np.ndarray
    extent: float
    arm_mouth: float
    segments: TSegments | None = None


#: Neutral name for the layout type both builders return.
MazeLayout = YMazeLayout


def _wall_box(
    p0: np.ndarray, p1: np.ndarray, thickness: float, height: float, extend: float = 0.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Box standing outside the edge ``p0 -> p1`` of a counter-clockwise outline.

    The box's inner face lies on the edge; the box occupies the right-hand side
    of the direction of travel (outward for a counter-clockwise outline).

    Args:
      p0, p1: edge end points, ``(2,)``.
      thickness: box thickness (perpendicular to the edge).
      height: full box height.
      extend: extra length added to *both* ends of the box, along the edge.

    Returns:
      ``(pos (3,), halfsize (3,), yaw)``.
    """
    edge = p1 - p0
    length = float(np.linalg.norm(edge))
    u = edge / length
    outward = np.array([u[1], -u[0]])
    centre = 0.5 * (p0 + p1) + 0.5 * thickness * outward
    pos = np.array([centre[0], centre[1], 0.5 * height])
    halfsize = np.array([0.5 * length + extend, 0.5 * thickness, 0.5 * height])
    return pos, halfsize, float(np.arctan2(u[1], u[0]))


def _walls_from_outline(
    outline: list[np.ndarray], cap_edges: set[int], thickness: float, height: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One `_wall_box` per edge of a closed counter-clockwise ``outline``.

    Edge ``i`` runs ``outline[i] -> outline[i + 1]``; the edges in
    ``cap_edges`` are widened by one wall thickness at both ends so they close
    the outer corners.

    Returns:
      ``(wall_pos (N, 3), wall_halfsize (N, 3), wall_yaw (N,))``.
    """
    pos, halfsize, yaw = [], [], []
    for i, p0 in enumerate(outline):
        p1 = outline[(i + 1) % len(outline)]
        b_pos, b_half, b_yaw = _wall_box(
            p0, p1, thickness, height, extend=thickness if i in cap_edges else 0.0
        )
        pos.append(b_pos)
        halfsize.append(b_half)
        yaw.append(b_yaw)
    return np.stack(pos), np.stack(halfsize), np.array(yaw)


def _bounding_extent(
    wall_pos: np.ndarray, wall_halfsize: np.ndarray, wall_yaw: np.ndarray
) -> float:
    """Half-width of the origin-centred bounding square of every box corner."""
    cos_y, sin_y = np.cos(wall_yaw), np.sin(wall_yaw)
    sx = np.array([-1.0, -1.0, 1.0, 1.0])
    sy = np.array([-1.0, 1.0, -1.0, 1.0])
    lx = sx[None, :] * wall_halfsize[:, 0:1]
    ly = sy[None, :] * wall_halfsize[:, 1:2]
    corner_x = wall_pos[:, 0:1] + cos_y[:, None] * lx - sin_y[:, None] * ly
    corner_y = wall_pos[:, 1:2] + sin_y[:, None] * lx + cos_y[:, None] * ly
    return float(max(np.abs(corner_x).max(), np.abs(corner_y).max()))


def build_y_maze(spec: YMazeSpec = YMazeSpec()) -> YMazeLayout:
    """Builds the Y-maze wall boxes and key points for ``spec``.

    Args:
      spec: maze dimensions.

    Returns:
      The `YMazeLayout`.

    Raises:
      ValueError: if the dimensions cannot form a Y (arm angle outside
        ``(0, 180)`` deg, non-positive stem, or arms too short to clear the
        junction).
    """
    if not 0.0 < spec.arm_angle_deg < 180.0:
        raise ValueError(
            f"arm_angle_deg must be in (0, 180), got {spec.arm_angle_deg}."
        )
    half = np.deg2rad(spec.arm_angle_deg) / 2.0  # each arm's angle from +y
    sin_h, cos_h = np.sin(half), np.cos(half)
    w = 0.5 * spec.corridor_width
    thick = spec.wall_thickness

    # Axial distance (along an arm axis) of the arm's mouth: where its lower
    # wall meets the stem wall.  The notch between the arms is at axial w/tan(h).
    mouth = w * np.tan(half / 2.0)
    notch_axial = w / np.tan(half)
    if (
        spec.stem_length <= 0.0
        or spec.arm_length <= max(mouth, notch_axial) + spec.treat_inset
    ):
        raise ValueError(
            f"stem_length ({spec.stem_length}) must be > 0 and arm_length ({spec.arm_length}) "
            f"must exceed the junction ({max(mouth, notch_axial):.3f}) plus treat_inset."
        )

    # Right arm (+x): axis, and the normal pointing to its lower/right side.
    d_r = np.array([sin_h, cos_h])
    n_r = np.array([cos_h, -sin_h])
    end_r = (spec.arm_length + w) * d_r  # inner face of the right arm's end cap

    def mirror(p: np.ndarray) -> np.ndarray:
        return np.array([-p[0], p[1]])

    p_stem_bl = np.array([-w, -spec.stem_length - w])
    p_stem_br = np.array([w, -spec.stem_length - w])
    # Stem right wall meets the right-arm lower wall (on x == w).
    p_mouth_r = mouth * d_r + w * n_r
    p_cap_r_lo = end_r + w * n_r
    p_cap_r_hi = end_r - w * n_r
    p_notch = np.array([0.0, w / sin_h])  # right/left arm upper walls meet on x == 0

    # Counter-clockwise outline; interior on the left of every edge.
    outline = [
        p_stem_bl,
        p_stem_br,
        p_mouth_r,
        p_cap_r_lo,
        p_cap_r_hi,
        p_notch,
        mirror(p_cap_r_hi),
        mirror(p_cap_r_lo),
        mirror(p_mouth_r),
    ]
    # Edge i runs outline[i] -> outline[i + 1]; the end caps are widened by one
    # wall thickness at both ends so they close the outer corners.
    wall_pos, wall_halfsize, wall_yaw = _walls_from_outline(
        outline, {0, 3, 6}, thick, spec.wall_height
    )

    # Arm 0 is the -x arm, arm 1 the +x arm.
    arm_dir = np.stack([mirror(d_r), d_r])
    junction_xy = np.zeros(2)
    arm_end_xy = junction_xy + spec.arm_length * arm_dir
    treat_xy = arm_end_xy - spec.treat_inset * arm_dir

    extent = _bounding_extent(wall_pos, wall_halfsize, wall_yaw)

    return YMazeLayout(
        wall_pos=wall_pos,
        wall_halfsize=wall_halfsize,
        wall_yaw=wall_yaw,
        junction_xy=junction_xy,
        start_xy=np.array([0.0, -spec.stem_length]),
        start_yaw=float(np.pi / 2.0),
        arm_dir=arm_dir,
        arm_end_xy=arm_end_xy,
        treat_xy=treat_xy,
        extent=extent,
        arm_mouth=float(mouth),
    )


def build_t_maze(spec: TMazeSpec = TMazeSpec()) -> YMazeLayout:
    """Builds the T-maze wall boxes, key points and segment table for ``spec``.

    See the module docstring for the layout.  Junction centre at the origin,
    stem along -y, arm 0 = left (-x), arm 1 = right (+x).

    Args:
      spec: maze dimensions (clear inner dimensions).

    Returns:
      A `YMazeLayout` (the env's one layout type) with ``segments`` set.

    Raises:
      ValueError: if a dimension is non-positive, or the arm is too short for
        its spout point to lie past the entry line
        (``arm_length <= spout_offset + entry_margin``).
    """
    if (
        spec.corridor_width <= 0.0
        or spec.segment_length <= 0.0
        or int(spec.n_stem_segments) < 1
        or spec.wall_thickness <= 0.0
        or spec.wall_height <= 0.0
        or spec.spout_offset < 0.0
        or spec.entry_margin < 0.0
    ):
        raise ValueError(f"Invalid T-maze dimensions: {spec}.")
    if spec.arm_length <= spec.spout_offset + spec.entry_margin:
        raise ValueError(
            f"arm_length ({spec.arm_length}) must exceed spout_offset + entry_margin "
            f"({spec.spout_offset} + {spec.entry_margin}): the spout must lie past "
            "the entry line."
        )
    n_seg = int(spec.n_stem_segments)
    w = 0.5 * spec.corridor_width
    back_y = -(w + n_seg * spec.segment_length)  # stem back wall's inner face
    reach = w + spec.arm_length  # |x| of the arm end walls' inner faces

    # Counter-clockwise outline; interior on the left of every edge.
    outline = [
        np.array([-w, back_y]),
        np.array([w, back_y]),
        np.array([w, -w]),  # stem right wall meets the right arm's lower wall
        np.array([reach, -w]),
        np.array([reach, w]),
        np.array([-reach, w]),  # the top wall spans both arms and the junction
        np.array([-reach, -w]),
        np.array([-w, -w]),
    ]
    # Caps: the stem back wall (0) and the two arm end walls (3, 5).
    wall_pos, wall_halfsize, wall_yaw = _walls_from_outline(
        outline, {0, 3, 5}, spec.wall_thickness, spec.wall_height
    )

    arm_dir = np.array([[-1.0, 0.0], [1.0, 0.0]])
    spout = reach - spec.spout_offset
    arm_end_xy = spout * arm_dir
    stem_edges_y = back_y + spec.segment_length * np.arange(n_seg + 1, dtype=float)
    stem_edges_y[-1] = -w  # exact, whatever the rounding of the sum above

    return YMazeLayout(
        wall_pos=wall_pos,
        wall_halfsize=wall_halfsize,
        wall_yaw=wall_yaw,
        junction_xy=np.zeros(2),
        start_xy=np.array([0.0, back_y + 0.5 * spec.segment_length]),
        start_yaw=float(np.pi / 2.0),
        arm_dir=arm_dir,
        arm_end_xy=arm_end_xy,
        treat_xy=arm_end_xy.copy(),
        extent=_bounding_extent(wall_pos, wall_halfsize, wall_yaw),
        arm_mouth=float(w + spec.entry_margin),
        segments=TSegments(
            stem_edges_y=stem_edges_y, half_width=float(w), arm_reach=float(reach)
        ),
    )


def locate(layout: YMazeLayout, xy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Locates a world xy position in the maze.  jnp-traceable (jit / vmap safe).

    The position is projected on each arm's axis.  It is *in* an arm once its
    axial coordinate passes ``layout.arm_mouth``; the arm with the larger axial
    coordinate wins (ties resolve to arm 0: in the Y only on the symmetry axis
    inside the solid wedge; in the T only at x = 0, which is never past the
    mouth).  Everywhere else (stem and junction) the arm id is ``-1``.

    Shape-agnostic: it reads only ``junction_xy``, ``arm_dir``, ``arm_end_xy``
    and ``arm_mouth``.  In the T maze an arm is reported exactly when
    ``|x| > corridor_width / 2 + entry_margin``, which in the free space only
    the arm row reaches.

    Args:
      layout: from `build_y_maze` or `build_t_maze`.  Its arrays are baked in
        as constants.
      xy: world position, shape ``(..., 2)``.

    Returns:
      ``(arm_id, dist_to_end)``: ``arm_id`` int32 in ``{-1, 0, 1}`` and
      ``dist_to_end`` float32, the axial distance from the position to its
      arm's ``arm_end_xy`` (Y: the arm end, negative past the end cap; T: the
      spout point, negative past it).  For ``arm_id == -1`` it is the same
      expression for the nearer arm, kept finite and continuous across the
      mouth, but it is not otherwise meaningful.
    """
    xy = jnp.asarray(xy, dtype=jnp.float32)
    junction = jnp.asarray(layout.junction_xy, dtype=jnp.float32)
    arm_dir = jnp.asarray(layout.arm_dir, dtype=jnp.float32)
    arm_len = jnp.sum(
        (jnp.asarray(layout.arm_end_xy, dtype=jnp.float32) - junction) * arm_dir,
        axis=-1,
    )
    axial = (xy - junction) @ arm_dir.T  # (..., 2): axial coordinate along each arm
    arm = jnp.argmax(axial, axis=-1)
    axial_best = jnp.max(axial, axis=-1)
    arm_id = jnp.where(axial_best > layout.arm_mouth, arm, -1).astype(jnp.int32)
    dist_to_end = (jnp.take(arm_len, arm) - axial_best).astype(jnp.float32)
    return arm_id, dist_to_end


def segment_of(layout: YMazeLayout, xy: jnp.ndarray) -> jnp.ndarray:
    """T-maze segment label of a world xy position.  jnp-traceable (jit / vmap safe).

    For the default T (`T_SEGMENT_NAMES`): 0 = ``start``, 1-4 = ``seg1`` -
    ``seg4``, 5 = ``junction``, 6 = ``arm_left``, 7 = ``arm_right``, and -1
    outside the clear outline.  In general, with ``n`` stem segments: stem
    segment ``k`` -> ``k``, junction -> ``n``, left / right arm -> ``n + 1`` /
    ``n + 2`` (names from `t_segment_names`).  Boundaries (all in the world
    frame, ``w`` = half the corridor width):

    * stem segment ``k``: ``|x| <= w`` and ``y`` in ``[stem_edges_y[k],
      stem_edges_y[k + 1])`` (the last edge is the junction edge ``-w``);
    * junction: ``|x| <= w`` and ``|y| <= w`` (closed);
    * arms: ``w < |x| <= arm_reach`` and ``|y| <= w``.

    A Y layout carries no segment table (``layout.segments is None``) and gets
    -1 everywhere.

    Args:
      layout: from `build_t_maze` (or `build_y_maze`, see above).  Baked in as
        constants.
      xy: world position, shape ``(..., 2)``.

    Returns:
      int32 labels, shape ``xy.shape[:-1]``.
    """
    xy = jnp.asarray(xy, dtype=jnp.float32)
    seg = layout.segments
    if seg is None:
        return jnp.full(xy.shape[:-1], -1, dtype=jnp.int32)
    edges = jnp.asarray(seg.stem_edges_y, dtype=jnp.float32)
    n = int(edges.shape[0]) - 1
    half = jnp.float32(seg.half_width)
    reach = jnp.float32(seg.arm_reach)
    x, y = xy[..., 0], xy[..., 1]
    ax, ay = jnp.abs(x), jnp.abs(y)
    in_row = (ay <= half) & (ax <= reach)
    junction = in_row & (ax <= half)
    in_stem = (ax <= half) & (y >= edges[0]) & (y < edges[n])
    # Number of interior boundaries at or below y: 0 .. n - 1.
    k = jnp.sum(y[..., None] >= edges[1:n], axis=-1).astype(jnp.int32)
    arm = jnp.where(x < 0.0, n + 1, n + 2).astype(jnp.int32)
    label = jnp.full(x.shape, -1, dtype=jnp.int32)
    label = jnp.where(in_stem, k, label)
    label = jnp.where(in_row, arm, label)
    label = jnp.where(junction, jnp.int32(n), label)
    return label.astype(jnp.int32)
