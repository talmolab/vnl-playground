"""Convert MuJoCo box geom to an equivalent mesh with UV texture coordinates."""

import numpy as np


def box_to_mesh_asset(
    half_extents: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create mesh vertices, faces, and UV coords equivalent to a box geom.

    Args:
        half_extents: (hx, hy, hz) half-sizes of the box.

    Returns:
        vertices: (24, 3) float64 -- 4 vertices per face, 6 faces.
        faces: (12, 3) int32 -- triangle indices.
        texcoords: (24, 2) float64 -- UV coordinates per vertex.
    """
    hx, hy, hz = half_extents

    # 6 faces, each with 4 corners (separate vertices for correct normals/UVs)
    # Order: +X, -X, +Y, -Y, +Z, -Z
    vertices = np.array(
        [
            # +X face
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [hx, hy, hz],
            [hx, -hy, hz],
            # -X face
            [-hx, hy, -hz],
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, hz],
            # +Y face
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, hy, hz],
            # -Y face
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [-hx, -hy, hz],
            # +Z face (top -- most visible)
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
            # -Z face (bottom)
            [-hx, hy, -hz],
            [hx, hy, -hz],
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
        ],
        dtype=np.float64,
    )

    # Two triangles per face
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # +X
            [4, 5, 6],
            [4, 6, 7],  # -X
            [8, 9, 10],
            [8, 10, 11],  # +Y
            [12, 13, 14],
            [12, 14, 15],  # -Y
            [16, 17, 18],
            [16, 18, 19],  # +Z
            [20, 21, 22],
            [20, 22, 23],  # -Z
        ],
        dtype=np.int32,
    )

    # UV: each face maps to full [0,1]x[0,1] texture space
    face_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    texcoords = np.tile(face_uv, (6, 1))  # (24, 2)

    return vertices, faces, texcoords
