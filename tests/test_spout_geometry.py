"""Unit tests for `vnl_playground.tasks.rodent.spout_geometry` (CPU, no GPU).

The Y-maze is built host-side with numpy; `locate` is the only jnp function and
must work under `jax.jit` / `jax.vmap`.  Wall boxes follow the MuJoCo convention
that the env will use: ``pos`` is the box centre, ``halfsize`` is
(half-length along the wall, half-thickness, half-height) and ``yaw`` rotates
the box about +z.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vnl_playground.tasks.rodent.spout_geometry import (
    YMazeLayout,
    YMazeSpec,
    build_y_maze,
    locate,
)

SPEC = YMazeSpec()
LAYOUT = build_y_maze(SPEC)


# ---------------------------------------------------------------------------
# helpers (2-D, rotated-box geometry)
# ---------------------------------------------------------------------------
def _to_box_frame(layout: YMazeLayout, i: int, pts: np.ndarray) -> np.ndarray:
    """Expresses world-frame xy points ``(..., 2)`` in wall ``i``'s frame."""
    c, s = np.cos(layout.wall_yaw[i]), np.sin(layout.wall_yaw[i])
    d = pts - layout.wall_pos[i, :2]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def _dist_to_walls(layout: YMazeLayout, pts: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest wall box (0 if inside one)."""
    best = np.full(pts.shape[:-1], np.inf)
    for i in range(len(layout.wall_yaw)):
        local = _to_box_frame(layout, i, pts)
        q = np.maximum(np.abs(local) - layout.wall_halfsize[i, :2], 0.0)
        best = np.minimum(best, np.hypot(q[..., 0], q[..., 1]))
    return best


def _ray_wall_distance(
    layout: YMazeLayout, origin: np.ndarray, dirs: np.ndarray
) -> np.ndarray:
    """Distance along each unit ray ``dirs (R, 2)`` to the nearest wall box (slab test)."""
    best = np.full(len(dirs), np.inf)
    for i in range(len(layout.wall_yaw)):
        c, s = np.cos(layout.wall_yaw[i]), np.sin(layout.wall_yaw[i])
        o = _to_box_frame(layout, i, origin)
        dl = np.stack(
            [c * dirs[:, 0] + s * dirs[:, 1], -s * dirs[:, 0] + c * dirs[:, 1]], -1
        )
        h = layout.wall_halfsize[i, :2]
        par = np.abs(dl) < 1e-12
        inv = 1.0 / np.where(par, 1.0, dl)
        ta, tb = (-h - o) * inv, (h - o) * inv
        inside_slab = np.abs(o) <= h
        t_lo = np.where(par, np.where(inside_slab, -np.inf, np.inf), np.minimum(ta, tb))
        t_hi = np.where(par, np.where(inside_slab, np.inf, -np.inf), np.maximum(ta, tb))
        t_near, t_far = t_lo.max(-1), t_hi.min(-1)
        hit = (t_near <= t_far) & (t_far >= 0.0)
        t = np.where(t_near >= 0.0, t_near, t_far)
        best = np.minimum(best, np.where(hit, t, np.inf))
    return best


def _centrelines(layout: YMazeLayout, n: int) -> list[np.ndarray]:
    """``n`` points along the stem, arm 0 and arm 1 centrelines, each ``(n, 2)``."""
    u = np.linspace(0.0, 1.0, n)[:, None]
    stem = layout.start_xy + u * (layout.junction_xy - layout.start_xy)
    arms = [
        layout.junction_xy + u * (layout.arm_end_xy[a] - layout.junction_xy)
        for a in range(2)
    ]
    return [stem, *arms]


# ---------------------------------------------------------------------------
# spec / layout structure
# ---------------------------------------------------------------------------
def test_spec_defaults_and_frozen():
    assert dataclasses.is_dataclass(SPEC)
    assert (
        SPEC.stem_length,
        SPEC.arm_length,
        SPEC.arm_angle_deg,
        SPEC.corridor_width,
        SPEC.wall_thickness,
        SPEC.wall_height,
        SPEC.appear_distance,
        SPEC.spout_radius,
        SPEC.treat_inset,
    ) == (0.9, 0.9, 120.0, 0.465, 0.15, 0.3, 0.5, 0.15, 0.15)
    with pytest.raises(dataclasses.FrozenInstanceError):
        SPEC.stem_length = 1.0  # type: ignore[misc]


def test_layout_shapes_and_wall_dims():
    lay = LAYOUT
    n = len(lay.wall_yaw)
    assert lay.wall_pos.shape == (n, 3) and lay.wall_halfsize.shape == (n, 3)
    assert lay.junction_xy.shape == (2,) and lay.start_xy.shape == (2,)
    assert lay.arm_dir.shape == (2, 2) and lay.arm_end_xy.shape == (2, 2)
    assert lay.treat_xy.shape == (2, 2)
    np.testing.assert_allclose(lay.junction_xy, [0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(lay.arm_dir, axis=1), 1.0, atol=1e-12)
    # Every wall is a thin, full-height slab standing on the floor.
    np.testing.assert_allclose(lay.wall_halfsize[:, 1], SPEC.wall_thickness / 2)
    np.testing.assert_allclose(lay.wall_halfsize[:, 2], SPEC.wall_height / 2)
    np.testing.assert_allclose(lay.wall_pos[:, 2], SPEC.wall_height / 2)


def test_arm_geometry_matches_spec():
    half = np.deg2rad(SPEC.arm_angle_deg) / 2
    # arms at +-60 deg from +y, arm 0 on the -x (left) side, arm 1 on +x.
    np.testing.assert_allclose(
        LAYOUT.arm_dir[0], [-np.sin(half), np.cos(half)], atol=1e-12
    )
    np.testing.assert_allclose(
        LAYOUT.arm_dir[1], [np.sin(half), np.cos(half)], atol=1e-12
    )
    np.testing.assert_allclose(
        LAYOUT.arm_end_xy,
        LAYOUT.junction_xy + SPEC.arm_length * LAYOUT.arm_dir,
        atol=1e-12,
    )
    # The stem runs along -y from the junction; the rat starts at its far end facing the junction.
    np.testing.assert_allclose(LAYOUT.start_xy, [0.0, -SPEC.stem_length], atol=1e-12)
    np.testing.assert_allclose(LAYOUT.start_yaw, np.pi / 2, atol=1e-12)


# ---------------------------------------------------------------------------
# core behaviour
# ---------------------------------------------------------------------------
def test_symmetric_arms():
    e = LAYOUT.arm_end_xy
    np.testing.assert_allclose(e[0], [-e[1, 0], e[1, 1]], atol=1e-6)


def test_centerlines_clear():
    # Clearance from a centreline to any wall box must be at least the (conservative)
    # corridor_width/2 - wall_thickness/2.
    floor = SPEC.corridor_width / 2 - SPEC.wall_thickness / 2 - 1e-3
    for line in _centrelines(LAYOUT, 200):
        assert _dist_to_walls(LAYOUT, line).min() >= floor


def test_enclosed():
    angles = (np.arange(64) + 0.5) * (2 * np.pi / 64)
    dirs = np.stack([np.cos(angles), np.sin(angles)], -1)
    for line in _centrelines(LAYOUT, 50):
        for p in line:
            assert np.all(_ray_wall_distance(LAYOUT, p, dirs) < 3.0), f"leak from {p}"


def test_treat_inside_arm():
    np.testing.assert_allclose(
        LAYOUT.treat_xy,
        LAYOUT.arm_end_xy - SPEC.treat_inset * LAYOUT.arm_dir,
        atol=1e-12,
    )


def test_locate():
    arm, _ = locate(LAYOUT, jnp.asarray(LAYOUT.start_xy))
    assert int(arm) == -1
    for a in range(2):
        arm, dist = locate(LAYOUT, jnp.asarray(LAYOUT.treat_xy[a]))
        assert int(arm) == a
        assert float(dist) == pytest.approx(SPEC.treat_inset, abs=1e-4)
        p = LAYOUT.arm_end_xy[a] - 0.4 * LAYOUT.arm_dir[a]
        arm, dist = locate(LAYOUT, jnp.asarray(p))
        assert int(arm) == a
        assert float(dist) == pytest.approx(0.4, abs=1e-4)


# ---------------------------------------------------------------------------
# additional checks
# ---------------------------------------------------------------------------
def test_clear_width_is_full_corridor_width():
    """No wall box intrudes into the corridors: the clear width is corridor_width, not less."""
    rng = np.random.default_rng(0)
    w = SPEC.corridor_width / 2
    margin = 1e-6
    # Stem corridor rectangle and the two arm rectangles, each including the alcove (half a
    # corridor width) behind its nominal end point.
    stem = np.stack(
        [
            rng.uniform(-w + margin, w - margin, 4000),
            rng.uniform(-SPEC.stem_length - w + margin, 0.0, 4000),
        ],
        -1,
    )
    pts = [stem]
    for a in range(2):
        d = LAYOUT.arm_dir[a]
        n = np.array([-d[1], d[0]])
        s = rng.uniform(
            0.0, SPEC.arm_length + w, 4000
        )  # incl. the alcove behind arm_end_xy
        lat = rng.uniform(-w + margin, w - margin, 4000)
        pts.append(s[:, None] * d + lat[:, None] * n)
    for p in pts:
        assert _dist_to_walls(LAYOUT, p).min() > 0.0, (
            "a wall box intrudes into the corridor"
        )


def test_junction_wedge_is_solid():
    """The wedge between the arms is closed: a point just above the notch is inside a wall."""
    w = SPEC.corridor_width / 2
    notch_y = w / np.sin(np.deg2rad(SPEC.arm_angle_deg) / 2)
    assert _dist_to_walls(LAYOUT, np.array([[0.0, notch_y + 0.02]]))[0] == 0.0
    # ... but just below it (junction free space) is clear.
    assert _dist_to_walls(LAYOUT, np.array([[0.0, notch_y - 0.02]]))[0] > 0.0


def test_end_points_have_clearance_for_the_body():
    # Rat ~0.31 m long: half length 0.155 m must fit behind the start point / past an arm end.
    ends = np.concatenate([LAYOUT.start_xy[None], LAYOUT.arm_end_xy])
    d = _dist_to_walls(LAYOUT, ends)
    np.testing.assert_allclose(d, SPEC.corridor_width / 2, atol=1e-9)
    assert SPEC.corridor_width / 2 > 0.155


def test_extent_bounds_all_walls_tightly():
    ext = 0.0
    for i in range(len(LAYOUT.wall_yaw)):
        c, s = np.cos(LAYOUT.wall_yaw[i]), np.sin(LAYOUT.wall_yaw[i])
        hx, hy = LAYOUT.wall_halfsize[i, :2]
        for sx in (-1, 1):
            for sy in (-1, 1):
                corner = LAYOUT.wall_pos[i, :2] + np.array(
                    [c * sx * hx - s * sy * hy, s * sx * hx + c * sy * hy]
                )
                ext = max(ext, np.abs(corner).max())
    assert LAYOUT.extent == pytest.approx(ext, abs=1e-9)


def test_locate_regions_and_continuity():
    w = SPEC.corridor_width / 2
    # Stem and junction -> -1 (including just inside the stem mouth).
    for p in ([0.0, -0.45], [w - 0.01, -0.5], [0.0, 0.0], [0.0, 0.2], [0.0, -0.1]):
        arm, _ = locate(LAYOUT, jnp.asarray(p))
        assert int(arm) == -1, p
    # Deep inside each arm -> that arm; the off-axis lateral position is irrelevant.
    for a in range(2):
        d = LAYOUT.arm_dir[a]
        n = np.array([-d[1], d[0]])
        for s in (0.3, 0.6, 0.85):
            for lat in (-0.1, 0.0, 0.1):
                p = s * d + lat * n
                arm, dist = locate(LAYOUT, jnp.asarray(p))
                assert int(arm) == a
                assert float(dist) == pytest.approx(SPEC.arm_length - s, abs=1e-4)


def test_locate_arm_boundary_is_the_corridor_mouth():
    """The arm begins at the mouth (``arm_mouth`` along the arm axis), not at the junction centre."""
    half = np.deg2rad(SPEC.arm_angle_deg) / 2
    mouth = SPEC.corridor_width / 2 * np.tan(half / 2)
    assert LAYOUT.arm_mouth == pytest.approx(mouth, abs=1e-12)
    for a in range(2):
        d = LAYOUT.arm_dir[a]
        before, _ = locate(LAYOUT, jnp.asarray((mouth - 0.01) * d))
        after, _ = locate(LAYOUT, jnp.asarray((mouth + 0.01) * d))
        assert int(before) == -1 and int(after) == a


def test_locate_is_jittable_vmappable_and_typed():
    jitted = jax.jit(lambda xy: locate(LAYOUT, xy))
    arm, dist = jitted(jnp.asarray(LAYOUT.treat_xy[1]))
    assert arm.dtype == jnp.int32 and dist.dtype == jnp.float32
    assert int(arm) == 1 and float(dist) == pytest.approx(SPEC.treat_inset, abs=1e-4)

    pts = jnp.asarray(
        np.concatenate(
            [
                LAYOUT.start_xy[None],
                LAYOUT.treat_xy,
                LAYOUT.arm_end_xy - 0.4 * LAYOUT.arm_dir,
            ]
        )
    )
    arms, dists = jax.jit(jax.vmap(lambda p: locate(LAYOUT, p)))(pts)
    np.testing.assert_array_equal(np.asarray(arms), [-1, 0, 1, 0, 1])
    np.testing.assert_allclose(np.asarray(dists)[1:], [0.15, 0.15, 0.4, 0.4], atol=1e-4)
    assert np.all(np.isfinite(np.asarray(dists)))


def test_locate_mirror_symmetry():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.0, 1.0, (200, 2))
    mirrored = pts * np.array([-1.0, 1.0])
    a0, d0 = jax.vmap(lambda p: locate(LAYOUT, p))(jnp.asarray(pts))
    a1, d1 = jax.vmap(lambda p: locate(LAYOUT, p))(jnp.asarray(mirrored))
    a0, a1 = np.asarray(a0), np.asarray(a1)
    np.testing.assert_allclose(np.asarray(d0), np.asarray(d1), atol=1e-5)
    in_arm = a0 >= 0
    np.testing.assert_array_equal(a1[in_arm], 1 - a0[in_arm])
    np.testing.assert_array_equal(a1[~in_arm], -1)


def test_invalid_spec_raises():
    with pytest.raises(ValueError):
        build_y_maze(YMazeSpec(arm_angle_deg=200.0))
    with pytest.raises(ValueError):
        build_y_maze(YMazeSpec(arm_length=0.1))


# ===========================================================================
# T maze (a mouse T maze at x2.5)
# ===========================================================================
from scipy import ndimage  # noqa: E402

from vnl_playground.tasks.rodent.spout_geometry import (  # noqa: E402
    T_SEGMENT_NAMES,
    MazeLayout,
    TMazeSpec,
    build_t_maze,
    segment_of,
    t_segment_names,
)

T_SPEC = TMazeSpec()
T_LAYOUT = build_t_maze(T_SPEC)
TW = 0.125  # half the clear corridor width
T_BACK_Y = -2.625  # the stem's back-wall inner face
T_END_X = 0.8125  # each arm's end-wall inner face, |x|
T_SPOUT_X = 0.7125
T_MOUTH = 0.155  # arm mouth (|x| = 0.125) + entry margin 0.03


def _in_t_outline(pts: np.ndarray, inset: float = 0.0) -> np.ndarray:
    """True where ``pts`` lie in the T's clear outline, shrunk by ``inset``."""
    x, y = pts[..., 0], pts[..., 1]
    stem = (np.abs(x) <= TW - inset) & (y >= T_BACK_Y + inset) & (y <= TW - inset)
    row = (np.abs(x) <= T_END_X - inset) & (np.abs(y) <= TW - inset)
    return stem | row


def _t_free_samples(n: int, seed: int, margin: float = 1e-6) -> np.ndarray:
    """Uniform points inside the stem rectangle and inside the arm row."""
    rng = np.random.default_rng(seed)
    stem = np.stack(
        [
            rng.uniform(-TW + margin, TW - margin, n),
            rng.uniform(T_BACK_Y + margin, -TW, n),
        ],
        -1,
    )
    row = np.stack(
        [
            rng.uniform(-T_END_X + margin, T_END_X - margin, n),
            rng.uniform(-TW + margin, TW - margin, n),
        ],
        -1,
    )
    return np.concatenate([stem, row])


def test_t_spec_defaults_and_frozen():
    assert dataclasses.is_dataclass(T_SPEC)
    assert (
        T_SPEC.corridor_width,
        T_SPEC.segment_length,
        T_SPEC.n_stem_segments,
        T_SPEC.arm_length,
        T_SPEC.wall_thickness,
        T_SPEC.wall_height,
        T_SPEC.spout_offset,
        T_SPEC.entry_margin,
    ) == (0.25, 0.50, 5, 0.6875, 0.05, 0.30, 0.10, 0.03)
    with pytest.raises(dataclasses.FrozenInstanceError):
        T_SPEC.arm_length = 1.0  # type: ignore[misc]


def test_t_layout_is_the_y_layout_type():
    """The env consumes one layout type; ``MazeLayout`` is a neutral alias for it."""
    assert MazeLayout is YMazeLayout
    assert isinstance(T_LAYOUT, YMazeLayout)
    lay = T_LAYOUT
    n = len(lay.wall_yaw)
    assert lay.wall_pos.shape == (n, 3) and lay.wall_halfsize.shape == (n, 3)
    assert lay.arm_dir.shape == (2, 2) and lay.arm_end_xy.shape == (2, 2)
    assert lay.treat_xy.shape == (2, 2)
    np.testing.assert_allclose(lay.junction_xy, [0.0, 0.0])
    # Arm 0 = left (-x), arm 1 = right (+x), as in the Y.
    np.testing.assert_allclose(lay.arm_dir, [[-1.0, 0.0], [1.0, 0.0]], atol=1e-12)
    # Thin, full-height walls standing on the floor.
    np.testing.assert_allclose(lay.wall_halfsize[:, 1], 0.05 / 2)
    np.testing.assert_allclose(lay.wall_halfsize[:, 2], 0.30 / 2)
    np.testing.assert_allclose(lay.wall_pos[:, 2], 0.30 / 2)
    # The Y layout carries no segment table.
    assert LAYOUT.segments is None and T_LAYOUT.segments is not None


def test_t_exact_clear_dimensions():
    """Corridor 0.25 everywhere, stem 2.50, arms 0.6875 each, footprint 1.625 x 2.75."""
    px, mx = np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]])
    py, my = np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]])

    def ray(origin, d):
        return float(_ray_wall_distance(T_LAYOUT, np.asarray(origin, float), d)[0])

    # Stem: clear width 0.25 at every height, centred on x = 0.
    for y in np.linspace(T_BACK_Y + 0.01, -TW - 0.01, 41):
        assert ray([0.0, y], px) == pytest.approx(TW, abs=1e-9), y
        assert ray([0.0, y], mx) == pytest.approx(TW, abs=1e-9), y
    # Arm row: clear width 0.25 along both arms, centred on y = 0.
    for x in np.concatenate(
        [
            np.linspace(-T_END_X + 0.01, -TW - 0.01, 21),
            np.linspace(TW + 0.01, T_END_X - 0.01, 21),
        ]
    ):
        assert ray([x, 0.0], py) == pytest.approx(TW, abs=1e-9), x
        assert ray([x, 0.0], my) == pytest.approx(TW, abs=1e-9), x
    # From the junction centre: top wall 0.125 up, back wall 2.625 down (stem 2.50 past the
    # junction edge), end walls 0.8125 to either side (arms 0.6875 past the junction edge).
    up, down = ray([0.0, 0.0], py), ray([0.0, 0.0], my)
    left, right = ray([0.0, 0.0], mx), ray([0.0, 0.0], px)
    assert up == pytest.approx(TW, abs=1e-9)
    assert down - TW == pytest.approx(2.50, abs=1e-9)
    assert left - TW == pytest.approx(0.6875, abs=1e-9)
    assert right - TW == pytest.approx(0.6875, abs=1e-9)
    assert left + right == pytest.approx(1.625, abs=1e-9)
    assert up + down == pytest.approx(2.75, abs=1e-9)
    # Five 0.50 stem segments.
    assert T_SPEC.n_stem_segments * T_SPEC.segment_length == pytest.approx(2.50)


@pytest.mark.parametrize("inflate", [0.0, 0.04])
def test_t_watertight_flood_fill(inflate):
    """Flood-fill from the start never leaves the clear outline (walls as built, and
    inflated by 0.04 = half the 0.08 m rat width), and still reaches both spouts."""
    h = 0.005
    xs = np.arange(-1.2, 1.2, h) + h / 2
    ys = np.arange(-3.0, 0.5, h) + h / 2
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx, gy], -1)
    free = _dist_to_walls(T_LAYOUT, pts) > inflate
    labels, _ = ndimage.label(free, structure=np.ones((3, 3), dtype=int))  # 8-connected

    def cell(p):
        return int(np.argmin(np.abs(xs - p[0]))), int(np.argmin(np.abs(ys - p[1])))

    start_label = labels[cell(T_LAYOUT.start_xy)]
    assert start_label > 0
    fill = labels == start_label
    # Bounded: the fill never touches the raster border ...
    assert not (
        fill[0].any() or fill[-1].any() or fill[:, 0].any() or fill[:, -1].any()
    )
    # ... and stays inside the clear outline (no leak through or between the walls).
    assert np.all(_in_t_outline(pts[fill]))
    # Both spouts are reachable from the start.
    for a in range(2):
        assert labels[cell(T_LAYOUT.treat_xy[a])] == start_label, a
    # Every free outline cell is reached (no sealed-off pocket).
    inside = _in_t_outline(pts, inset=inflate + h)
    assert np.all(fill[inside])


def test_t_no_intrusion_and_walls_flush():
    """No wall box intrudes into the clear outline, and every wall's inner face lies on it."""
    pts = _t_free_samples(5000, seed=3)
    assert _dist_to_walls(T_LAYOUT, pts).min() > 0.0, "a wall box intrudes into the T"
    # Just outside every outline edge (and around every corner) is solid wall.
    outline = np.array(
        [
            [-TW, T_BACK_Y],
            [TW, T_BACK_Y],
            [TW, -TW],
            [T_END_X, -TW],
            [T_END_X, TW],
            [-T_END_X, TW],
            [-T_END_X, -TW],
            [-TW, -TW],
        ]
    )
    for eps in (1e-4, 0.025, 0.05 - 1e-4):
        ring = []
        for i in range(len(outline)):
            p0, p1 = outline[i], outline[(i + 1) % len(outline)]
            u = (p1 - p0) / np.linalg.norm(p1 - p0)
            out = np.array([u[1], -u[0]])  # counter-clockwise outline: right = outward
            s = np.linspace(0.0, 1.0, 200)[:, None]
            ring.append(p0 + s * (p1 - p0) + eps * out)
        ring = np.concatenate(ring)
        # Concave corners: the offset edge points run inside the clear outline there.
        ring = ring[~_in_t_outline(ring)]
        assert np.all(_dist_to_walls(T_LAYOUT, ring) == 0.0), eps


def test_t_start_pose():
    np.testing.assert_allclose(T_LAYOUT.start_xy, [0.0, -2.375], atol=1e-12)
    assert T_LAYOUT.start_yaw == pytest.approx(
        np.pi / 2, abs=1e-12
    )  # facing +y, the junction
    # The centre of the `start` segment: 0.25 clear behind it.
    back = _ray_wall_distance(T_LAYOUT, T_LAYOUT.start_xy, np.array([[0.0, -1.0]]))[0]
    assert back == pytest.approx(0.25, abs=1e-9)


def test_t_spout_points():
    np.testing.assert_allclose(
        T_LAYOUT.arm_end_xy, [[-T_SPOUT_X, 0.0], [T_SPOUT_X, 0.0]], atol=1e-12
    )
    np.testing.assert_array_equal(T_LAYOUT.treat_xy, T_LAYOUT.arm_end_xy)
    for a in range(2):
        # 0.10 in front of the end wall's inner face, on the arm's centreline.
        d = _ray_wall_distance(
            T_LAYOUT, T_LAYOUT.arm_end_xy[a], T_LAYOUT.arm_dir[a][None]
        )[0]
        assert d == pytest.approx(0.10, abs=1e-9)


def test_t_extent_bounds_all_walls_tightly():
    ext = 0.0
    for i in range(len(T_LAYOUT.wall_yaw)):
        c, s = np.cos(T_LAYOUT.wall_yaw[i]), np.sin(T_LAYOUT.wall_yaw[i])
        hx, hy = T_LAYOUT.wall_halfsize[i, :2]
        for sx in (-1, 1):
            for sy in (-1, 1):
                corner = T_LAYOUT.wall_pos[i, :2] + np.array(
                    [c * sx * hx - s * sy * hy, s * sx * hx + c * sy * hy]
                )
                ext = max(ext, np.abs(corner).max())
    assert T_LAYOUT.extent == pytest.approx(ext, abs=1e-9)
    assert T_LAYOUT.extent == pytest.approx(2.625 + 0.05, abs=1e-9)


def test_t_locate_regions():
    assert T_LAYOUT.arm_mouth == pytest.approx(T_MOUTH, abs=1e-12)
    # Stem, junction and the entry margin (0.125 < |x| <= 0.155): no arm.
    for p in (
        [0.0, -2.6],
        [0.12, -2.375],
        [-0.12, -1.0],
        [0.0, -0.2],
        [0.0, 0.0],
        [0.12, 0.12],
        [-0.12, -0.12],
        [0.15, 0.0],
        [-0.15, 0.1],
        [0.15, -0.1],
    ):
        arm, _ = locate(T_LAYOUT, jnp.asarray(p))
        assert int(arm) == -1, p
    # Past the margin, at the spout and past it: that arm, dist = 0.7125 - |x| (negative past).
    for a, sign in ((0, -1.0), (1, 1.0)):
        for ax in (0.16, 0.3, 0.4125, T_SPOUT_X, 0.78, 0.81):
            for y in (-0.1, 0.0, 0.1):
                arm, dist = locate(T_LAYOUT, jnp.asarray([sign * ax, y]))
                assert int(arm) == a, (a, ax, y)
                assert float(dist) == pytest.approx(T_SPOUT_X - ax, abs=1e-5), (
                    a,
                    ax,
                    y,
                )
    _, dist = locate(T_LAYOUT, jnp.asarray([0.78, 0.0]))
    assert float(dist) < 0.0


def test_t_locate_on_free_space():
    """Everywhere in the T: an arm exactly when |x| > 0.155 (only the arm row reaches that),
    and every in-arm distance is within the T config's appear_distance (0.70).

    ``locate`` projects with a matmul, and on GPU JAX's DEFAULT f32 matmul precision is
    TF32, so there its distances carry up to ~2.5e-4 m of error (the Y's too; left as is
    so the Y stays bit-identical).  The exact semantics are checked at "highest" precision
    and the default-precision deviation is bounded separately.
    """
    pts = _t_free_samples(4000, seed=5)
    x = pts[:, 0].astype(np.float32)
    expect = np.where(np.abs(x) > np.float32(T_MOUTH), np.where(x < 0, 0, 1), -1)
    with jax.default_matmul_precision("highest"):
        exact = jax.jit(jax.vmap(lambda p: locate(T_LAYOUT, p)))
        arm, dist = exact(jnp.asarray(pts, jnp.float32))
    arm, dist = np.asarray(arm), np.asarray(dist)
    np.testing.assert_array_equal(arm, expect)
    assert np.all(np.abs(pts[arm >= 0, 1]) <= TW)  # only inside the arm row
    assert dist[arm >= 0].max() <= 0.70
    np.testing.assert_allclose(
        dist[arm >= 0], T_SPOUT_X - np.abs(pts[arm >= 0, 0]), atol=1e-5
    )

    # Default precision (what the env runs): same arm away from the entry line, distances
    # within 1e-3 m.
    arm_d, dist_d = jax.jit(jax.vmap(lambda p: locate(T_LAYOUT, p)))(
        jnp.asarray(pts, jnp.float32)
    )
    arm_d, dist_d = np.asarray(arm_d), np.asarray(dist_d)
    clear = np.abs(np.abs(x) - T_MOUTH) > 1e-3
    np.testing.assert_array_equal(arm_d[clear], expect[clear])
    np.testing.assert_allclose(dist_d, dist, atol=1e-3)


def test_t_segment_names():
    assert T_SEGMENT_NAMES == (
        "start",
        "seg1",
        "seg2",
        "seg3",
        "seg4",
        "junction",
        "arm_left",
        "arm_right",
    )
    assert t_segment_names(5) == T_SEGMENT_NAMES
    assert t_segment_names(3) == (
        "start",
        "seg1",
        "seg2",
        "junction",
        "arm_left",
        "arm_right",
    )


# (point, expected segment id) at every boundary and inside every region.
_E = 1e-4
_SEGMENT_CASES = (
    # region interiors
    [([0.0, -2.375 + 0.5 * k], k) for k in range(5)]
    + [([0.0, 0.0], 5), ([-0.46875, 0.0], 6), ([0.46875, 0.0], 7)]
    # stem boundaries in y: segment k covers [-2.625 + 0.5k, -2.125 + 0.5k)
    + [([0.0, T_BACK_Y], 0), ([0.0, T_BACK_Y - _E], -1)]
    + [([0.0, T_BACK_Y + 0.5 * k], k) for k in range(1, 5)]
    + [([0.0, T_BACK_Y + 0.5 * k - _E], k - 1) for k in range(1, 5)]
    # stem -> junction at y = -0.125 (closed junction)
    + [([0.0, -TW], 5), ([0.0, -TW - _E], 4), ([TW, -TW], 5), ([-TW, -TW - _E], 4)]
    # stem side walls
    + [([TW, -1.0], 3), ([-TW, -1.0], 3), ([TW + _E, -1.0], -1), ([-TW - _E, -1.0], -1)]
    # junction <-> arms at |x| = 0.125 (closed junction), and the row's top / bottom faces
    + [([TW, 0.0], 5), ([-TW, 0.0], 5), ([TW + _E, 0.0], 7), ([-TW - _E, 0.0], 6)]
    + [
        ([0.0, TW], 5),
        ([0.0, TW + _E], -1),
        ([0.5, TW], 7),
        ([0.5, -TW], 7),
        ([-0.5, TW], 6),
    ]
    + [([0.5, TW + _E], -1), ([0.5, -TW - _E], -1), ([-0.5, -TW - _E], -1)]
    # arm end walls
    + [
        ([T_END_X, 0.0], 7),
        ([-T_END_X, 0.0], 6),
        ([T_END_X + _E, 0.0], -1),
        ([-T_END_X - _E, 0.0], -1),
    ]
    # well outside
    + [([2.0, 2.0], -1), ([0.0, 1.0], -1), ([0.0, -3.0], -1), ([0.5, -1.0], -1)]
)


def test_t_segment_of_regions_and_boundaries():
    pts = jnp.asarray([p for p, _ in _SEGMENT_CASES], jnp.float32)
    expect = np.array([s for _, s in _SEGMENT_CASES])
    eager = np.array([int(segment_of(T_LAYOUT, p)) for p in pts])
    np.testing.assert_array_equal(eager, expect)
    batched = segment_of(T_LAYOUT, pts)  # (..., 2) batch, no vmap
    assert batched.dtype == jnp.int32 and batched.shape == (len(expect),)
    np.testing.assert_array_equal(np.asarray(batched), expect)
    vj = jax.jit(jax.vmap(lambda p: segment_of(T_LAYOUT, p)))(pts)
    assert vj.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(vj), expect)
    # Arbitrary leading batch dims.
    grid = segment_of(T_LAYOUT, pts.reshape(-1, 1, 2))
    np.testing.assert_array_equal(np.asarray(grid).reshape(-1), expect)


def test_t_segment_of_agrees_with_locate():
    pts = jnp.asarray(_t_free_samples(4000, seed=7), jnp.float32)
    seg = np.asarray(jax.jit(jax.vmap(lambda p: segment_of(T_LAYOUT, p)))(pts))
    arm = np.asarray(jax.vmap(lambda p: locate(T_LAYOUT, p)[0])(pts))
    assert np.all(seg >= 0), "a free-space point fell outside every segment"
    np.testing.assert_array_equal(seg[arm == 0], 6)
    np.testing.assert_array_equal(seg[arm == 1], 7)
    assert np.all(arm[seg <= 5] == -1)
    # Every segment is populated.
    assert set(np.unique(seg)) == set(range(8))


def test_t_segment_of_other_stem_count():
    lay = build_t_maze(TMazeSpec(n_stem_segments=3))
    np.testing.assert_allclose(
        lay.start_xy, [0.0, -1.375]
    )  # back wall -1.625, segment 0 centre
    cases = {
        (0.0, -1.375): 0,
        (0.0, -0.875): 1,
        (0.0, -0.375): 2,
        (0.0, 0.0): 3,
        (-0.5, 0.0): 4,
        (0.5, 0.0): 5,
    }
    pts = jnp.asarray(list(cases), jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(segment_of(lay, pts)), list(cases.values())
    )


def test_segment_of_y_layout_is_minus_one():
    """Y layouts carry no segment table: ``segment_of`` is -1 everywhere (documented)."""
    pts = jnp.asarray(
        np.random.default_rng(2).uniform(-1.5, 1.5, (500, 2)), jnp.float32
    )
    seg = jax.jit(lambda p: segment_of(LAYOUT, p))(pts)
    assert seg.dtype == jnp.int32 and seg.shape == (500,)
    assert np.all(np.asarray(seg) == -1)
    assert int(segment_of(LAYOUT, jnp.zeros(2))) == -1


def test_t_invalid_spec_raises():
    with pytest.raises(ValueError):
        build_t_maze(TMazeSpec(corridor_width=0.0))
    with pytest.raises(ValueError):
        build_t_maze(TMazeSpec(n_stem_segments=0))
    with pytest.raises(ValueError):
        build_t_maze(TMazeSpec(segment_length=-0.5))
    with pytest.raises(ValueError):  # the spout must lie past the entry line
        build_t_maze(TMazeSpec(arm_length=0.12))
    with pytest.raises(ValueError):
        build_t_maze(TMazeSpec(wall_thickness=0.0))
