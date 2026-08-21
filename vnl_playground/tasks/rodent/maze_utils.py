# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Host-side maze helpers for the maze-forage task.

PURE NUMPY.  There is deliberately no JAX in this module: everything here runs
once, host-side, in an environment's ``__init__`` (maze generation, wall
covering, arena geometry).  Nothing here is ever called under ``jax.jit``.

The greedy rectangle-covering code (``GridCoordinates``, ``MazeWall``,
``_MazeWallCoveringContext``, ``make_walls``) is **adapted from dm_control**,
``dm_control/locomotion/arenas/covering.py`` (Apache-2.0, header kept above).
The port is faithful; the only changes are typing/docstrings and accepting a
plain numpy character grid instead of requiring a ``labmaze.TextGrid``.

``grid_to_world`` / ``world_to_grid`` follow the coordinate convention of
``dm_control/locomotion/arenas/mazes.py`` (``grid_to_world_positions`` /
``world_to_grid_positions``, lines ~357-372): the grid is centred on the world
origin, ``+x`` follows increasing column and ``+y`` follows *decreasing* row
(i.e. row 0 is the "north" edge of the arena).
"""

from __future__ import annotations

import collections
import collections.abc
from typing import Sequence, Tuple, Union

import numpy as np

__all__ = [
    "WALL_CHAR",
    "OPEN_CHAR",
    "GridCoordinates",
    "MazeWall",
    "as_char_grid",
    "grid_to_text",
    "make_walls",
    "generate_maze",
    "grid_to_world",
    "world_to_grid",
    "free_cell_indices",
    "free_cells",
    "wall_boxes",
    "maze_extent",
]

WALL_CHAR = "*"
OPEN_CHAR = " "

GridCoordinates = collections.namedtuple("GridCoordinates", ("y", "x"))
MazeWall = collections.namedtuple("MazeWall", ("start", "end"))

# A maze may be given as a char grid, a labmaze.TextGrid, a newline separated
# string, a sequence of row strings, or a boolean/int occupancy array.
MazeLike = Union[np.ndarray, str, Sequence[str]]


# ---------------------------------------------------------------------------
# Grid representation helpers
# ---------------------------------------------------------------------------
def as_char_grid(
    maze: MazeLike,
    wall_char: str = WALL_CHAR,
    open_char: str = OPEN_CHAR,
) -> np.ndarray:
    """Normalises any supported maze representation to a 2-D ``<U1`` array.

    Args:
      maze: one of
        * a 2-D numpy array of single characters (incl. ``labmaze.TextGrid``),
        * a boolean or integer 2-D array where truthy == wall,
        * a newline-separated string,
        * a sequence of equal-length row strings.
      wall_char: character to emit for wall cells when converting a binary grid.
      open_char: character to emit for open cells when converting a binary grid.

    Returns:
      A ``(height, width)`` array of dtype ``<U1``.  ``labmaze.TextGrid`` inputs
      are returned as a plain ``np.ndarray`` view (it is an ndarray subclass).
    """
    if isinstance(maze, str):
        rows = maze.splitlines()
        rows = [r for r in rows if r]
        return _rows_to_grid(rows)

    if isinstance(maze, np.ndarray):
        arr = np.asarray(maze)
        if arr.ndim != 2:
            raise ValueError(f"maze array must be 2-D, got shape {arr.shape}.")
        if arr.dtype.kind == "S":
            arr = np.char.decode(arr)
        if arr.dtype.kind == "U":
            if arr.dtype.itemsize // 4 == 1:
                return arr.view(np.ndarray)
            # Wider string dtype: re-cast so every entry is a single char.
            return np.asarray(arr, dtype="<U1")
        if arr.dtype.kind in ("b", "i", "u", "f"):
            return np.where(arr.astype(bool), wall_char, open_char).astype("<U1")
        raise ValueError(f"Unsupported maze dtype {arr.dtype!r}.")

    if isinstance(maze, collections.abc.Sequence):
        rows = list(maze)
        if rows and all(isinstance(r, str) for r in rows):
            return _rows_to_grid(rows)
        return as_char_grid(np.asarray(maze), wall_char, open_char)

    raise TypeError(f"Cannot interpret {type(maze)!r} as a maze grid.")


def _rows_to_grid(rows: Sequence[str]) -> np.ndarray:
    if not rows:
        raise ValueError("Empty maze.")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("All maze rows must have the same length.")
    return np.array([list(r) for r in rows], dtype="<U1")


def grid_to_text(maze: MazeLike) -> str:
    """Renders a maze grid back to a newline-separated string (trailing '\\n')."""
    grid = as_char_grid(maze)
    return "".join("".join(row) + "\n" for row in grid)


# ---------------------------------------------------------------------------
# Greedy rectangular wall covering -- adapted from dm_control's covering.py
# ---------------------------------------------------------------------------
class _MazeWallCoveringContext:
    """Calculates a covering of text mazes with rectangular walls.

    Adapted verbatim (modulo style) from dm_control
    ``locomotion/arenas/covering.py``.

    This class uses a greedy algorithm to try and minimize the number of geoms
    generated to create a given maze. The solution is not guaranteed to be
    optimal, but in most cases should result in a significantly smaller number
    of geoms than if each cell were treated as an individual box.
    """

    def __init__(self, text_maze, wall_char=WALL_CHAR, make_odd_sized_walls=False):
        """Initializes this _MazeWallCoveringContext.

        Args:
          text_maze: A 2-D character grid (``labmaze.TextGrid`` or ndarray).
          wall_char: (optional) The character that signifies a wall.
          make_odd_sized_walls: (optional) A boolean, if `True` all wall sections
            generated span odd numbers of grid cells. This option exists
            primarily to appease MuJoCo's texture repeating algorithm.
        """
        self._text_maze = text_maze
        self._wall_char = wall_char
        self._make_odd_sized_walls = make_odd_sized_walls
        self._covered = np.full(text_maze.shape, False, dtype=bool)
        self._maze_size = GridCoordinates(*text_maze.shape)
        self._next_start = GridCoordinates(0, 0)
        self._calculated = False
        self._walls = ()

    def calculate(self) -> Tuple[MazeWall, ...]:
        """Calculates a covering of the text maze with rectangular walls.

        Returns:
          A tuple of `MazeWall` objects, each describing the corners of a wall.
        """
        if not self._calculated:
            self._calculated = True
            self._find_next_start()
            walls = []
            while self._next_start.y < self._maze_size.y:
                walls.append(self._find_next_wall())
                self._find_next_start()
            self._walls = tuple(walls)
        return self._walls

    def _find_next_start(self) -> None:
        """Moves `self._next_start` to the top-left corner of the next wall."""
        for y in range(self._next_start.y, self._maze_size.y):
            start_x = self._next_start.x if y == self._next_start.y else 0
            for x in range(start_x, self._maze_size.x):
                if self._text_maze[y, x] == self._wall_char and not self._covered[y, x]:
                    self._next_start = GridCoordinates(y, x)
                    return
        self._next_start = self._maze_size

    def _scan_row(self, row: int, start_col: int, end_col: int) -> int:
        """Scans a row of text maze to find the longest strip of wall."""
        for col in range(start_col, end_col):
            if self._text_maze[row, col] != self._wall_char or self._covered[row, col]:
                return col
        return end_col

    def _find_next_wall(self) -> MazeWall:
        """Finds the largest piece of rectangular wall at the current location.

        This function assumes that `self._next_start` is already at the top-left
        corner of the next piece of wall.

        Returns:
          A `MazeWall` named tuple representing the next piece of wall created.
        """
        start = self._next_start
        x = self._maze_size.x
        end_x_for_rows = []
        total_cells = []

        for y in range(start.y, self._maze_size.y):
            x = self._scan_row(y, start.x, x)
            if x > start.x:
                if self._make_odd_sized_walls and (x - start.x) % 2 == 0:
                    x -= 1
                end_x_for_rows.append(x)
                total_cells.append((x - start.x) * (y - start.y + 1))
            else:
                break

        if not self._make_odd_sized_walls:
            end_y_offset = total_cells.index(max(total_cells))
        else:
            end_y_offset = 2 * total_cells[::2].index(max(total_cells[::2]))
        end = GridCoordinates(
            start.y + end_y_offset + 1, end_x_for_rows[end_y_offset]
        )
        self._covered[start.y : end.y, start.x : end.x] = True
        self._next_start = GridCoordinates(start.y, end.x)
        return MazeWall(start, end)


def make_walls(
    text_maze: MazeLike,
    wall_char: str = WALL_CHAR,
    make_odd_sized_walls: bool = False,
) -> Tuple[MazeWall, ...]:
    """Calculates a covering of a text maze with rectangular walls.

    Adapted from dm_control ``locomotion/arenas/covering.py``.  Rectangles are
    half-open: a wall covers rows ``[start.y, end.y)`` and columns
    ``[start.x, end.x)``.  Every wall cell is covered by exactly one rectangle
    and no rectangle touches a floor cell.

    Args:
      text_maze: Anything accepted by `as_char_grid` (a `labmaze.TextGrid`, a
        character array, a binary array, a string, or a list of row strings).
      wall_char: (optional) The character that signifies a wall.
      make_odd_sized_walls: (optional) A boolean, if `True` all wall sections
        generated span odd numbers of grid cells. This option exists primarily
        to appease MuJoCo's texture repeating algorithm.

    Returns:
      A tuple of `MazeWall` objects, each describing the corners of a wall.
    """
    grid = as_char_grid(text_maze, wall_char=wall_char)
    context = _MazeWallCoveringContext(
        grid, wall_char=wall_char, make_odd_sized_walls=make_odd_sized_walls
    )
    return context.calculate()


# ---------------------------------------------------------------------------
# Maze generation (seeded, deterministic, numpy-only)
# ---------------------------------------------------------------------------
def generate_maze(
    maze_cells: Union[int, Tuple[int, int]] = 3,
    seed: int = 0,
    wall_char: str = WALL_CHAR,
    open_char: str = OPEN_CHAR,
    loop_fraction: float = 0.0,
) -> np.ndarray:
    """Generates a perfect maze with a seeded randomised-DFS ("recursive
    backtracker") carve.

    The returned grid is the standard "walls on even indices" layout: logical
    cell ``(i, j)`` lives at grid index ``(2 * i + 1, 2 * j + 1)`` and the grid
    is fully enclosed by a wall border.

    Args:
      maze_cells: number of logical cells, either an int (square) or an
        ``(n_rows, n_cols)`` pair.  The resulting grid is
        ``(2 * n_rows + 1, 2 * n_cols + 1)``, e.g. ``maze_cells=3`` -> 7x7.
      seed: RNG seed.  The same seed always produces the same maze.
      wall_char: character used for wall cells.
      open_char: character used for open cells.
      loop_fraction: (optional) in ``[0, 1]``.  Fraction of the remaining
        interior walls to knock out *after* carving, turning the tree-shaped
        perfect maze into one with loops.  ``0.0`` leaves a perfect maze.

    Returns:
      A ``(2 * n_rows + 1, 2 * n_cols + 1)`` array of dtype ``<U1``.
    """
    if isinstance(maze_cells, (int, np.integer)):
        n_rows = n_cols = int(maze_cells)
    else:
        n_rows, n_cols = (int(v) for v in maze_cells)
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"maze_cells must be >= 1, got {maze_cells!r}.")
    if not 0.0 <= loop_fraction <= 1.0:
        raise ValueError(f"loop_fraction must be in [0, 1], got {loop_fraction}.")

    rng = np.random.default_rng(seed)
    height, width = 2 * n_rows + 1, 2 * n_cols + 1
    grid = np.full((height, width), wall_char, dtype="<U1")

    visited = np.zeros((n_rows, n_cols), dtype=bool)
    start = (int(rng.integers(n_rows)), int(rng.integers(n_cols)))
    visited[start] = True
    grid[2 * start[0] + 1, 2 * start[1] + 1] = open_char
    stack = [start]
    # (drow, dcol) in logical-cell space.
    neighbour_offsets = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])

    while stack:
        i, j = stack[-1]
        candidates = []
        for di, dj in neighbour_offsets:
            ni, nj = i + int(di), j + int(dj)
            if 0 <= ni < n_rows and 0 <= nj < n_cols and not visited[ni, nj]:
                candidates.append((ni, nj))
        if not candidates:
            stack.pop()
            continue
        ni, nj = candidates[int(rng.integers(len(candidates)))]
        visited[ni, nj] = True
        # Open the neighbour cell and the wall between it and the current cell.
        grid[2 * ni + 1, 2 * nj + 1] = open_char
        grid[i + ni + 1, j + nj + 1] = open_char
        stack.append((ni, nj))

    if loop_fraction > 0.0:
        removable = _removable_interior_walls(grid, wall_char)
        n_remove = int(round(loop_fraction * len(removable)))
        if n_remove > 0 and len(removable) > 0:
            picks = rng.choice(len(removable), size=n_remove, replace=False)
            for p in np.atleast_1d(picks):
                r, c = removable[int(p)]
                grid[r, c] = open_char

    return grid


def _removable_interior_walls(grid: np.ndarray, wall_char: str) -> np.ndarray:
    """Interior wall cells that sit between two logical cells (not pillars)."""
    height, width = grid.shape
    out = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if grid[r, c] != wall_char:
                continue
            # Exactly one of (row, col) is odd for a between-cells wall; the
            # even/even positions are structural pillars and stay put.
            if (r % 2 == 1) == (c % 2 == 1):
                continue
            out.append((r, c))
    return np.array(out, dtype=int).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Grid <-> world coordinates (dm_control mazes.py convention)
# ---------------------------------------------------------------------------
def _offsets(grid_shape: Tuple[int, int]) -> Tuple[float, float]:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    return (height - 1) / 2.0, (width - 1) / 2.0  # (y_offset, x_offset)


def grid_to_world(
    grid_rc,
    cell_size: float,
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """Converts grid ``(row, col)`` indices to world ``(x, y)`` centres.

    Matches dm_control ``MazeWithTargets.grid_to_world_positions``: the grid is
    centred on the world origin, ``x`` grows with the column index and ``y``
    *shrinks* with the row index.

    Args:
      grid_rc: ``(2,)`` or ``(..., 2)`` array-like of ``(row, col)``.  Values may
        be fractional (e.g. rectangle midpoints).
      cell_size: world size of one grid cell, in metres.
      grid_shape: ``(height, width)`` of the full grid.

    Returns:
      Float array of world ``(x, y)`` with the same leading shape as `grid_rc`.
    """
    rc = np.asarray(grid_rc, dtype=float)
    if rc.shape[-1] != 2:
        raise ValueError(f"grid_rc last axis must be 2, got shape {rc.shape}.")
    y_offset, x_offset = _offsets(grid_shape)
    x = (rc[..., 1] - x_offset) * cell_size
    y = -(rc[..., 0] - y_offset) * cell_size
    return np.stack([x, y], axis=-1)


def world_to_grid(
    xy,
    cell_size: float,
    grid_shape: Tuple[int, int],
    as_int: bool = False,
) -> np.ndarray:
    """Inverse of `grid_to_world`: world ``(x, y)`` -> grid ``(row, col)``.

    Args:
      xy: ``(2,)`` or ``(..., 2)`` array-like of world ``(x, y)``.  A trailing
        z component is *not* accepted; slice it off first.
      cell_size: world size of one grid cell, in metres.
      grid_shape: ``(height, width)`` of the full grid.
      as_int: if True, round to the nearest cell and return an int array.
        Results are *not* clipped to the grid; do that yourself if needed.

    Returns:
      Array of ``(row, col)``, float unless `as_int`.
    """
    world = np.asarray(xy, dtype=float)
    if world.shape[-1] != 2:
        raise ValueError(f"xy last axis must be 2, got shape {world.shape}.")
    y_offset, x_offset = _offsets(grid_shape)
    row = y_offset - world[..., 1] / cell_size
    col = x_offset + world[..., 0] / cell_size
    rc = np.stack([row, col], axis=-1)
    if as_int:
        return np.rint(rc).astype(int)
    return rc


def maze_extent(cell_size: float, grid_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Half-extent ``(x, y)`` of the maze footprint in metres (centred on 0)."""
    height, width = int(grid_shape[0]), int(grid_shape[1])
    return width * cell_size / 2.0, height * cell_size / 2.0


# ---------------------------------------------------------------------------
# Open-cell queries
# ---------------------------------------------------------------------------
def free_cell_indices(maze: MazeLike, wall_char: str = WALL_CHAR) -> np.ndarray:
    """``(M, 2)`` int array of ``(row, col)`` indices of non-wall cells."""
    grid = as_char_grid(maze, wall_char=wall_char)
    rows, cols = np.nonzero(grid != wall_char)
    return np.stack([rows, cols], axis=-1).astype(int)


def free_cells(
    maze: MazeLike,
    cell_size: float = 1.0,
    wall_char: str = WALL_CHAR,
) -> np.ndarray:
    """``(M, 2)`` float array of world ``(x, y)`` centres of the open cells.

    Used for sampling spawn and treat positions at ``reset()`` (host-side: the
    array is built once and then indexed with a traced index under jit).
    """
    grid = as_char_grid(maze, wall_char=wall_char)
    return grid_to_world(free_cell_indices(grid, wall_char), cell_size, grid.shape)


# ---------------------------------------------------------------------------
# Arena geometry
# ---------------------------------------------------------------------------
def wall_boxes(
    walls: Sequence[MazeWall],
    cell_size: float,
    grid_shape: Tuple[int, int],
    wall_height: float,
    z_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts covering rectangles to MuJoCo box ``pos`` / ``size`` arrays.

    Ports the geometry of dm_control ``MazeWithTargets._make_wall_geoms``: each
    rectangle becomes one axis-aligned box that exactly fills its grid cells,
    sitting on the floor plane at ``z = z_offset``.

    Args:
      walls: rectangles from `make_walls`.
      cell_size: world size of one grid cell, in metres.
      grid_shape: ``(height, width)`` of the full grid.
      wall_height: full height of the walls, in metres.
      z_offset: world z of the floor the walls stand on.

    Returns:
      ``(pos, size)``, each ``(n_walls, 3)``.  ``size`` is MuJoCo's half-extent
      convention.
    """
    n = len(walls)
    pos = np.zeros((n, 3), dtype=float)
    size = np.zeros((n, 3), dtype=float)
    for i, wall in enumerate(walls):
        mid_y = (wall.start.y + wall.end.y - 1) / 2.0
        mid_x = (wall.start.x + wall.end.x - 1) / 2.0
        xy = grid_to_world((mid_y, mid_x), cell_size, grid_shape)
        pos[i] = (xy[0], xy[1], z_offset + wall_height / 2.0)
        size[i] = (
            (wall.end.x - mid_x - 0.5) * cell_size,
            (wall.end.y - mid_y - 0.5) * cell_size,
            wall_height / 2.0,
        )
    return pos, size
