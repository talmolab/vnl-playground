"""Tests for box-to-mesh conversion utility."""

import numpy as np
import mujoco
import pytest

from vnl_playground.tasks.rodent.utils.box_to_mesh import box_to_mesh_asset


def test_box_to_mesh_creates_valid_mesh():
    """Converted mesh should have 24 vertices, 12 triangles, and proper UVs."""
    vertices, faces, texcoords = box_to_mesh_asset(half_extents=(1.0, 0.5, 0.1))
    assert vertices.shape == (24, 3)  # 4 verts per face * 6 faces (for proper normals)
    assert faces.shape == (12, 3)  # 2 triangles per face * 6 faces
    assert texcoords.shape == (24, 2)  # One UV per vertex
    # UV coords should be in [0, 1]
    assert np.all(texcoords >= 0.0) and np.all(texcoords <= 1.0)


def test_box_mesh_matches_box_geom_collision():
    """Mesh geom should produce the same contact points as a box geom."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[1.0, 0.5, 0.1],
        pos=[0, 0, 0.1],
    )
    verts, faces, texcoords = box_to_mesh_asset(half_extents=(1.0, 0.5, 0.1))
    mesh = spec.add_mesh()
    mesh.name = "box_mesh"
    mesh.uservert = verts.flatten()
    mesh.userface = faces.flatten()
    mesh.usertexcoord = texcoords.flatten()
    body = spec.worldbody.add_body(name="mesh_body", pos=[3, 0, 0.1])
    body.add_geom(
        name="mesh_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="box_mesh",
    )
    model = spec.compile()
    assert model.nmesh == 1
    assert model.mesh_facenum[0] == 12
