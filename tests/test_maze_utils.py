"""Unit tests for `vnl_playground.tasks.rodent.maze_utils` (pure numpy, no GPU)."""

import numpy as np
import pytest

from vnl_playground.tasks.rodent.maze_utils import (
    OPEN_CHAR,
    WALL_CHAR,
    MazeWall,
    as_char_grid,
    free_cell_indices,
    free_cells,
    generate_maze,
    grid_to_text,
    grid_to_world,
    make_walls,
    maze_extent,
    wall_boxes,
    world_to_grid,
)

# A few hand-written mazes exercising L/T/plus shapes, ragged edges, and the
# degenerate all-wall / no-wall cases.
HAND_MAZES = [
    [
        "*****",
        "*   *",
        "*   *",
        "*   *",
        "*****",
    ],
    [
        "*******",
        "*  *  *",
        "*  *  *",
        "*******",
        "*     *",
        "*  ****",
        "*******",
    ],
    [
        "  ***  ",
        " ***** ",
        "***  **",
        "* * * *",
        "**  ***",
        " ***** ",
        "  ***  ",
    ],
    ["***", "***", "***"],  # all wall
    ["   ", "   ", "   "],  # no wall
    ["*"],
    [" "],
]

GEN_SEEDS = [0, 1, 2, 7, 42, 12345]


def _all_mazes():
    """Hand-written mazes plus generated ones of a few sizes/seeds."""
    for rows in HAND_MAZES:
        yield as_char_grid(rows)
    for seed in GEN_SEEDS:
        for cells in (1, 2, 3, 5, (2, 6)):
            yield generate_maze(cells, seed=seed)
        yield generate_maze(4, seed=seed, loop_fraction=0.3)


def _coverage_count(grid, walls):
    """Number of rectangles covering each grid cell."""
    count = np.zeros(grid.shape, dtype=int)
    for wall in walls:
        count[wall.start.y : wall.end.y, wall.start.x : wall.end.x] += 1
    return count


# ---------------------------------------------------------------------------
# make_walls -- the dm_control covering port
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("odd", [False, True])
def test_every_wall_cell_covered_exactly_once(odd):
    for grid in _all_mazes():
        walls = make_walls(grid, make_odd_sized_walls=odd)
        count = _coverage_count(grid, walls)
        is_wall = grid == WALL_CHAR
        assert np.all(count[is_wall] == 1), (
            f"wall cells not covered exactly once (odd={odd}):\n{grid_to_text(grid)}"
            f"\ncounts:\n{count}"
        )


@pytest.mark.parametrize("odd", [False, True])
def test_no_rectangle_touches_a_floor_cell(odd):
    for grid in _all_mazes():
        walls = make_walls(grid, make_odd_sized_walls=odd)
        count = _coverage_count(grid, walls)
        is_floor = grid != WALL_CHAR
        assert np.all(count[is_floor] == 0), (
            f"a rectangle covers a floor cell (odd={odd}):\n{grid_to_text(grid)}"
        )


@pytest.mark.parametrize("odd", [False, True])
def test_rectangles_are_pairwise_non_overlapping(odd):
    for grid in _all_mazes():
        walls = make_walls(grid, make_odd_sized_walls=odd)
        for i in range(len(walls)):
            a = walls[i]
            # Well-formed, non-empty, in-bounds half-open rectangle.
            assert a.end.y > a.start.y and a.end.x > a.start.x
            assert a.start.y >= 0 and a.start.x >= 0
            assert a.end.y <= grid.shape[0] and a.end.x <= grid.shape[1]
            for j in range(i + 1, len(walls)):
                b = walls[j]
                overlap_y = min(a.end.y, b.end.y) - max(a.start.y, b.start.y)
                overlap_x = min(a.end.x, b.end.x) - max(a.start.x, b.start.x)
                assert overlap_y <= 0 or overlap_x <= 0, (
                    f"rectangles {a} and {b} overlap (odd={odd})"
                )


def test_make_odd_sized_walls_spans_are_odd():
    for grid in _all_mazes():
        for wall in make_walls(grid, make_odd_sized_walls=True):
            assert (wall.end.y - wall.start.y) % 2 == 1
            assert (wall.end.x - wall.start.x) % 2 == 1


def test_covering_is_fewer_geoms_than_per_cell():
    grid = generate_maze(5, seed=0)
    walls = make_walls(grid)
    n_wall_cells = int(np.sum(grid == WALL_CHAR))
    assert 0 < len(walls) < n_wall_cells


def test_no_walls_returns_empty_tuple():
    walls = make_walls(["   ", "   "])
    assert walls == ()


def test_custom_wall_char():
    grid = ["#.#", ".#.", "#.#"]
    walls = make_walls(grid, wall_char="#")
    count = _coverage_count(as_char_grid(grid), walls)
    assert np.array_equal(count, np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))


def test_known_covering_is_a_single_rectangle():
    walls = make_walls(["***", "***"])
    assert len(walls) == 1
    assert walls[0] == MazeWall((0, 0), (2, 3))


def test_accepts_string_and_binary_grids():
    rows = ["***", "* *", "***"]
    from_rows = make_walls(rows)
    from_string = make_walls("\n".join(rows))
    binary = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    from_binary = make_walls(binary)
    assert from_rows == from_string == from_binary


def test_accepts_labmaze_text_grid():
    labmaze = pytest.importorskip("labmaze")
    maze = labmaze.RandomMaze(height=9, width=9, random_seed=3)
    grid = maze.entity_layer
    walls = make_walls(grid)
    count = _coverage_count(np.asarray(grid), walls)
    assert np.all(count[np.asarray(grid) == WALL_CHAR] == 1)
    assert np.all(count[np.asarray(grid) != WALL_CHAR] == 0)


# ---------------------------------------------------------------------------
# grid <-> world
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell_size", [0.35, 1.0, 2.5])
@pytest.mark.parametrize("shape", [(7, 7), (5, 11), (1, 1), (4, 6)])
def test_grid_world_round_trip_on_all_cells(cell_size, shape):
    rows, cols = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    rc = np.stack([rows.ravel(), cols.ravel()], axis=-1).astype(float)

    xy = grid_to_world(rc, cell_size, shape)
    assert xy.shape == rc.shape

    back = world_to_grid(xy, cell_size, shape)
    np.testing.assert_allclose(back, rc, atol=1e-9)
    np.testing.assert_array_equal(
        world_to_grid(xy, cell_size, shape, as_int=True), rc.astype(int)
    )


def test_grid_to_world_single_point_shape_and_convention():
    shape = (7, 7)
    centre = grid_to_world((3, 3), 0.35, shape)
    assert centre.shape == (2,)
    np.testing.assert_allclose(centre, [0.0, 0.0], atol=1e-12)

    # +col -> +x, +row -> -y (dm_control mazes.py convention).
    np.testing.assert_allclose(grid_to_world((3, 4), 0.35, shape), [0.35, 0.0])
    np.testing.assert_allclose(grid_to_world((4, 3), 0.35, shape), [0.0, -0.35])


def test_world_to_grid_round_trip_from_world_side():
    shape = (9, 5)
    cell_size = 0.35
    rng = np.random.default_rng(0)
    x_half, y_half = maze_extent(cell_size, shape)
    xy = np.stack(
        [
            rng.uniform(-x_half, x_half, size=64),
            rng.uniform(-y_half, y_half, size=64),
        ],
        axis=-1,
    )
    rc = world_to_grid(xy, cell_size, shape)
    np.testing.assert_allclose(grid_to_world(rc, cell_size, shape), xy, atol=1e-9)


def test_maze_extent_matches_grid_span():
    shape = (7, 9)
    cell_size = 0.35
    x_half, y_half = maze_extent(cell_size, shape)
    corners = grid_to_world(
        [[0, 0], [shape[0] - 1, shape[1] - 1]], cell_size, shape
    )
    # Cell centres sit half a cell inside the footprint edge.
    np.testing.assert_allclose(abs(corners[0][1]), y_half - cell_size / 2)
    np.testing.assert_allclose(abs(corners[1][0]), x_half - cell_size / 2)


def test_grid_to_world_rejects_bad_shape():
    with pytest.raises(ValueError):
        grid_to_world([1.0, 2.0, 3.0], 0.35, (7, 7))
    with pytest.raises(ValueError):
        world_to_grid(np.zeros((4, 3)), 0.35, (7, 7))


# ---------------------------------------------------------------------------
# maze generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cells", [1, 2, 3, 5, (2, 6)])
def test_generate_maze_shape_and_alphabet(cells):
    grid = generate_maze(cells, seed=0)
    n_rows, n_cols = (cells, cells) if isinstance(cells, int) else cells
    assert grid.shape == (2 * n_rows + 1, 2 * n_cols + 1)
    assert set(np.unique(grid)) <= {WALL_CHAR, OPEN_CHAR}
    # Border is fully walled and every logical cell is open.
    assert np.all(grid[0, :] == WALL_CHAR) and np.all(grid[-1, :] == WALL_CHAR)
    assert np.all(grid[:, 0] == WALL_CHAR) and np.all(grid[:, -1] == WALL_CHAR)
    assert np.all(grid[1::2, 1::2] == OPEN_CHAR)


@pytest.mark.parametrize("seed", GEN_SEEDS)
def test_generate_maze_is_deterministic_given_a_seed(seed):
    a = generate_maze(4, seed=seed)
    b = generate_maze(4, seed=seed)
    assert np.array_equal(a, b)
    # ... including the loop-carving branch.
    c = generate_maze(4, seed=seed, loop_fraction=0.4)
    d = generate_maze(4, seed=seed, loop_fraction=0.4)
    assert np.array_equal(c, d)


def test_generate_maze_differs_across_seeds():
    mazes = [generate_maze(5, seed=s) for s in GEN_SEEDS]
    texts = {grid_to_text(m) for m in mazes}
    assert len(texts) == len(mazes), "different seeds produced identical mazes"


def test_generate_maze_is_connected():
    """Perfect maze: every logical cell reachable from every other."""
    for seed in GEN_SEEDS:
        grid = generate_maze(5, seed=seed)
        open_mask = grid != WALL_CHAR
        start = tuple(free_cell_indices(grid)[0])
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                    and open_mask[nr, nc]
                    and (nr, nc) not in seen
                ):
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        assert len(seen) == int(open_mask.sum())


def test_loop_fraction_opens_more_cells_and_keeps_pillars():
    perfect = generate_maze(5, seed=0, loop_fraction=0.0)
    braided = generate_maze(5, seed=0, loop_fraction=0.5)
    assert int((braided != WALL_CHAR).sum()) > int((perfect != WALL_CHAR).sum())
    # Structural pillars (even row AND even col) are never removed.
    assert np.all(braided[2:-2:2, 2:-2:2] == WALL_CHAR)


def test_generate_maze_rejects_bad_args():
    with pytest.raises(ValueError):
        generate_maze(0, seed=0)
    with pytest.raises(ValueError):
        generate_maze(3, seed=0, loop_fraction=1.5)


# ---------------------------------------------------------------------------
# free cells
# ---------------------------------------------------------------------------
def test_free_cells_count_and_openness():
    cell_size = 0.35
    for grid in _all_mazes():
        idx = free_cell_indices(grid)
        xy = free_cells(grid, cell_size)
        n_open = int(np.sum(grid != WALL_CHAR))

        assert idx.shape == (n_open, 2)
        assert xy.shape == (n_open, 2)
        assert xy.dtype == np.float64

        if n_open == 0:
            continue
        # Every returned index is an open cell...
        assert np.all(grid[idx[:, 0], idx[:, 1]] != WALL_CHAR)
        # ... and no open cell is missing.
        assert {tuple(v) for v in idx} == {
            tuple(v) for v in np.argwhere(grid != WALL_CHAR)
        }
        # World coords map back onto exactly those cells.
        back = world_to_grid(xy, cell_size, grid.shape, as_int=True)
        np.testing.assert_array_equal(back, idx)
        assert np.all(grid[back[:, 0], back[:, 1]] != WALL_CHAR)


def test_free_cells_never_returns_a_wall_cell():
    grid = generate_maze(5, seed=11)
    xy = free_cells(grid, 0.35)
    rc = world_to_grid(xy, 0.35, grid.shape, as_int=True)
    assert not np.any(grid[rc[:, 0], rc[:, 1]] == WALL_CHAR)


def test_free_cells_on_all_wall_maze_is_empty():
    xy = free_cells(["***", "***"], 0.35)
    assert xy.shape == (0, 2)


# ---------------------------------------------------------------------------
# wall boxes (arena geometry)
# ---------------------------------------------------------------------------
def test_wall_boxes_cover_exactly_the_wall_cells():
    cell_size = 0.35
    wall_height = 0.3
    for grid in _all_mazes():
        walls = make_walls(grid)
        pos, size = wall_boxes(walls, cell_size, grid.shape, wall_height)
        assert pos.shape == (len(walls), 3)
        assert size.shape == (len(walls), 3)
        if len(walls) == 0:
            continue
        np.testing.assert_allclose(pos[:, 2], wall_height / 2)
        np.testing.assert_allclose(size[:, 2], wall_height / 2)
        assert np.all(size[:, :2] > 0)

        # Each box's footprint area equals its rectangle's cell area.
        for wall, p, s in zip(walls, pos, size):
            n_cells_x = wall.end.x - wall.start.x
            n_cells_y = wall.end.y - wall.start.y
            np.testing.assert_allclose(2 * s[0], n_cells_x * cell_size)
            np.testing.assert_allclose(2 * s[1], n_cells_y * cell_size)
            # Box centre == world centre of the rectangle's cell block.
            mid = ((wall.start.y + wall.end.y - 1) / 2, (wall.start.x + wall.end.x - 1) / 2)
            np.testing.assert_allclose(p[:2], grid_to_world(mid, cell_size, grid.shape))

        # No box overlaps any open cell centre.
        open_xy = free_cells(grid, cell_size)
        for p, s in zip(pos, size):
            inside = np.all(np.abs(open_xy - p[:2]) < s[:2], axis=-1)
            assert not np.any(inside)


# ---------------------------------------------------------------------------
# representation helpers
# ---------------------------------------------------------------------------
def test_as_char_grid_and_grid_to_text_round_trip():
    rows = ["***", "* *", "***"]
    grid = as_char_grid(rows)
    assert grid.shape == (3, 3)
    assert grid.dtype == np.dtype("<U1")
    assert grid_to_text(grid) == "***\n* *\n***\n"
    assert np.array_equal(as_char_grid(grid_to_text(grid)), grid)


def test_as_char_grid_rejects_ragged_and_bad_input():
    with pytest.raises(ValueError):
        as_char_grid(["***", "**"])
    with pytest.raises(ValueError):
        as_char_grid(np.zeros((2, 2, 2)))
    with pytest.raises(TypeError):
        as_char_grid(3.14)


# ---------------------------------------------------------------------------
# Faithfulness of the port
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("odd", [False, True])
def test_covering_matches_dm_control_exactly(odd):
    """Our port must return bit-identical rectangles to the dm_control original."""
    dm_covering = pytest.importorskip("dm_control.locomotion.arenas.covering")
    for grid in _all_mazes():
        ours = make_walls(grid, make_odd_sized_walls=odd)
        theirs = dm_covering.make_walls(grid, make_odd_sized_walls=odd)
        assert tuple(map(tuple, ours)) == tuple(map(tuple, theirs)), (
            f"covering diverged from dm_control (odd={odd}):\n{grid_to_text(grid)}"
        )
