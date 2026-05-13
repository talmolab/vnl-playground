"""Sanity-check consts shape for the new mesh stick model."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")


def test_stick_xml_path_points_to_mesh_fast():
    from vnl_playground.tasks.stick import consts
    assert consts.STICK_XML_PATH.name == "stick_mesh_fast.xml"
    assert consts.STICK_XML_PATH.is_file()


def test_stick_box_xml_path_kept_available():
    from vnl_playground.tasks.stick import consts
    assert consts.STICK_BOX_XML_PATH.name == "stick_fast.xml"
    assert consts.STICK_BOX_XML_PATH.is_file()


def test_joints_includes_thorax_and_total_41():
    from vnl_playground.tasks.stick import consts
    assert len(consts.JOINTS) == 41
    # The three thorax hinges that were commented out in the box model
    assert "04-t1-l" in consts.JOINTS
    assert "05-t2-l" in consts.JOINTS
    assert "06-t3-l" in consts.JOINTS


def test_foot_geoms_are_six_claw_collide_names():
    from vnl_playground.tasks.stick import consts
    assert set(consts.FOOT_GEOMS) == {
        "claw_collide_fl", "claw_collide_ml", "claw_collide_hl",
        "claw_collide_fr", "claw_collide_mr", "claw_collide_hr",
    }


def test_bodies_unchanged_count():
    from vnl_playground.tasks.stick import consts
    # 1 ref base + 3 thorax + 8 abdomen + 30 leg = 42
    assert len(consts.BODIES) == 42


def test_get_assets_includes_mesh_obj_and_png():
    """The stick base.get_assets() must bundle OBJ + texture under
    stick_insect_urdf/meshes/obj/ so the mesh XML can be loaded from bytes."""
    from vnl_playground.tasks.stick.base import get_assets
    assets = get_assets()
    obj_names = [k for k in assets if k.endswith(".obj")]
    png_names = [k for k in assets if k.endswith(".png")]
    mtl_names = [k for k in assets if k.endswith(".mtl")]
    assert len(obj_names) >= 86, f"expected >= 86 OBJ assets, got {len(obj_names)}"
    assert len(png_names) >= 1, f"expected a texture PNG, got {png_names}"
    assert len(mtl_names) >= 1, f"expected an MTL file, got {mtl_names}"
