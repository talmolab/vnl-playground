"""Measure what the *policy* actually sees in ``MazeForageVision``.

``mujoco.Renderer`` third-person figures are not evidence about the observation:
they apply a headlight and a skybox, neither of which exists in the Warp
ray-tracer that fills the ``vision`` observation.  Every number below was
measured through :class:`JaxVisionRenderer` -- the same class
``train_highlvl`` wraps the env with -- on an RTX 5090, ``mujoco 3.9.0`` /
``warp 1.12.1``, at the config default 64x64 grayscale, batch 6, ``maze_seed=0``.

Facts this script exists to keep honest (all re-derivable with ``--what all``):

1. **The Warp ray-tracer never computes UVs for a BOX geom.**
   ``mujoco_warp/_src/render.py::sample_texture`` only fills ``uv`` for
   ``GeomType.PLANE`` (local xy, in **metres**) and ``GeomType.MESH``
   (barycentric interpolation of the mesh's own texcoords).  A box falls through
   both branches with ``uv = (0, 0)``, so the whole geom is shaded with the
   *single* texel at the texture's origin -- measured within-wall std of
   **exactly 0.00000** for every box variant tried.

2. **That one texel still multiplies the base colour**, so a texture makes a box
   *darker*, never more detailed.  The shipped wall material is a builtin
   checker with ``rgb1=[0.30, 0.30, 0.34]``; walls render at 0.077 with it and
   0.252 without it -- a **3.3x brightness loss** bought for zero structure.

3. **Vertical faces get almost nothing from an overhead light.**  Shading is
   ``0.5 * base * ambient + sum_lights base * max(0, N.L)`` with
   ``ambient = (0.4,0.4,0.45)*h + (0.1,0.1,0.12)*(1-h)``, ``h = 0.5*(n_z+1)``.
   A wall has ``n_z = 0`` so ``h = 0.5`` and the ambient term is only
   ``0.125 * base``; both lights the env ships point down, so ``N.L ~ 0``.

4. **Most material and light attributes are simply not read by the kernel.**
   Measured ``mean|diff| == 0.0`` for material ``emission`` 0.0 vs 1.0,
   material ``texuniform`` 0 vs 1, and light ``diffuse`` 0.05 vs 1.0.  The only
   levers are ``mat_rgba``, ``mat_texid[mjTEXROLE_RGB]``, ``mat_texrepeat``,
   ``geom_rgba``, and light ``type`` / ``pos`` / ``dir`` / ``active`` /
   ``castshadow``.

5. **There is no skybox.**  On a ray miss the kernel returns early and the pixel
   keeps ``RenderContext.background_color``, hardcoded to ``(0.1, 0.1, 0.2)``
   in ``create_render_context`` (grayscale 0.1114) and *frozen* -- assigning to
   it raises ``FrozenInstanceError``.  Compiling ``go_to_target``'s
   ``outdoor_natural`` skybox into this model changes the policy pixels by
   ``mean|diff| = 0.0``.

The fix this script measures is :func:`add_wall_skins`: keep the collision
boxes exactly as they are but demote them to geom group 3 (outside the render
context's ``enabled_geom_groups=[0, 1, 2]`` **and** outside ``MjvOption``'s
default ``geomgroup=[1,1,1,0,0,0]``, so they vanish from both renderers), and
add one visual-only 24-vertex box mesh per wall whose UVs are baked in world
units so texel density is identical on every wall regardless of its size.

Usage::

    MUJOCO_GL=egl PYTHONPATH=<clone> python scripts/probe_maze_vision.py --what all
    MUJOCO_GL=egl PYTHONPATH=<clone> python scripts/probe_maze_vision.py \
        --what variants --out maze_vision_figs
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import copy

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from vnl_playground.tasks.rodent.maze_forage_vision import (
    MazeForageVision,
    default_config,
)
from vnl_playground.tasks.rodent.vision_jax import JaxVisionRenderer

#: Prefix for the visual-only wall meshes.  Deliberately NOT ``maze_wall_``:
#: the geometry tests filter on ``startswith("maze_wall_")`` to rasterise
#: corridor widths off the compiled boxes, and a skin picked up by that filter
#: would silently corrupt those measurements.
SKIN_PREFIX = "maze_wallskin_"

#: Geom group the collision boxes are demoted to.  Outside both
#: ``enabled_geom_groups=[0,1,2]`` (Warp) and ``MjvOption.geomgroup``'s default
#: ``[1,1,1,0,0,0]`` (mujoco.Renderer), so nothing double-draws.
COLLISION_ONLY_GROUP = 3

#: Face -> the four corner sign-triples, wound outward.
_BOX_FACES = {
    (0, +1): [(+1, -1, -1), (+1, +1, -1), (+1, +1, +1), (+1, -1, +1)],
    (0, -1): [(-1, +1, -1), (-1, -1, -1), (-1, -1, +1), (-1, +1, +1)],
    (1, +1): [(+1, +1, -1), (-1, +1, -1), (-1, +1, +1), (+1, +1, +1)],
    (1, -1): [(-1, -1, -1), (+1, -1, -1), (+1, -1, +1), (-1, -1, +1)],
    (2, +1): [(-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1)],
    (2, -1): [(-1, +1, -1), (+1, +1, -1), (+1, -1, -1), (-1, -1, -1)],
}


def box_skin_mesh(half, tile):
    """A 24-vertex box with UVs baked in world units.

    Each face gets its own four vertices so its UVs are independent, and the UV
    span of a face is ``world_extent / tile``.  With ``mat_texrepeat = (1, 1)``
    that makes one texture tile exactly ``tile`` metres on **every** wall --
    which a single shared unit-UV mesh cannot do: measured over the 17 walls of
    the default maze, unit UVs + a scalar ``texrepeat`` give world tiles from
    3.8 cm to 25 cm, a **7x** spread across the same scene.

    Args:
        half: ``(3,)`` box half-extents in metres.
        tile: World size of one texture repeat, in metres.

    Returns:
        ``(verts (24,3), faces (12,3), texcoords (24,2))``.
    """
    half = np.asarray(half, dtype=float)
    verts, faces, uvs = [], [], []
    for (axis, _sign), corners in _BOX_FACES.items():
        u_ax, v_ax = [a for a in (0, 1, 2) if a != axis]
        du, dv = 2 * half[u_ax] / tile, 2 * half[v_ax] / tile
        base = len(verts)
        for c in corners:
            verts.append((np.asarray(c, dtype=float) * half).tolist())
            uvs.append([(c[u_ax] * 0.5 + 0.5) * du, (c[v_ax] * 0.5 + 0.5) * dv])
        faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.asarray(verts), np.asarray(faces, dtype=int), np.asarray(uvs)


def add_wall_skins(spec, tile=0.10, rgb1=(0.35, 0.35, 0.35), rgb2=(0.95, 0.95, 0.95),
                   wall_rgba=(1.0, 1.0, 1.0, 1.0), texture_file=None):
    """Give every ``maze_wall_*`` box a textured, visual-only mesh skin.

    Mutates ``spec`` in place.  The collision boxes keep their geometry,
    ``contype`` and ``conaffinity`` untouched -- only ``group`` and ``material``
    change -- so the simulation is unaffected (measured: an 8-world 30-step
    rollout diverges by 1.5e-05, against a 1.7e-05 box-vs-box run-to-run floor).

    Args:
        spec: The env's ``MjSpec``, before ``compile()``.
        tile: World size of one texture repeat, in metres.
        rgb1, rgb2: Builtin-checker colours, used when ``texture_file`` is None.
        wall_rgba: Multiplies the sampled texel; the wall brightness knob.
        texture_file: Absolute path to a texture image (e.g. a labmaze wall
            asset).  Passed as an **absolute** path on purpose -- ``spec`` has a
            single ``compiler.texturedir`` and the outdoor-natural aesthetic
            already owns it, so a relative name here would resolve against the
            wrong directory.

    Returns:
        The list of skin geom names that were added.
    """
    for tex in list(spec.textures):
        if tex.name == "maze_wall_tex":
            spec.delete(tex)
    if texture_file is not None:
        spec.add_texture(name="maze_wall_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         file=str(texture_file))
    else:
        spec.add_texture(name="maze_wall_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         width=64, height=64, rgb1=list(rgb1), rgb2=list(rgb2))
    mat = spec.material("maze_wall_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "maze_wall_tex"
    mat.texrepeat = [1, 1]
    mat.rgba = list(wall_rgba)

    names = []
    for geom in [g for g in spec.worldbody.geoms if g.name.startswith("maze_wall_")]:
        verts, faces, uvs = box_skin_mesh(geom.size, tile)
        mesh = spec.add_mesh(name=f"{SKIN_PREFIX}{geom.name}_mesh")
        mesh.uservert = verts.reshape(-1).tolist()
        mesh.userface = faces.reshape(-1).tolist()
        mesh.usertexcoord = uvs.reshape(-1).tolist()
        name = f"{SKIN_PREFIX}{geom.name}"
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name, pos=list(geom.pos), material="maze_wall_mat",
            contype=0, conaffinity=0, group=0)
        names.append(name)
        geom.group = COLLISION_ONLY_GROUP
        geom.material = ""
    return names


def set_four_directional_lights(spec, tilt=1.0, castshadow=False):
    """Replace every light with four directional lights from +-x / +-y.

    ``dir = (+-1, 0, -tilt)`` and ``(0, +-1, -tilt)``.  Light ``diffuse`` is
    ignored by the kernel, so ``tilt`` is the only intensity knob there is: it
    splits a fixed budget between vertical faces (``N.L = 1/|dir|``) and the
    floor (``N.L = tilt/|dir|``, times four because all four lights reach it).

    Measured wall / floor grayscale at ``wall_rgba = 1.0``:
    ``tilt 0.6 -> 0.633 / 0.528``, ``tilt 1.0 -> 0.538 / 0.695``,
    ``tilt 1.5 -> 0.438 / 0.780``.  ``tilt 0.35`` clips 33.6% of the frame.
    """
    for light in list(spec.lights):
        spec.delete(light)
    for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        spec.worldbody.add_light(
            name=f"maze_dir_light_{i}", pos=[0, 0, 4], dir=[dx, dy, -tilt],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=int(castshadow))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def reset_qpos(env, batch=6, seed=0):
    """``jax.vmap``'d reset -> ``(batch, nq)``; every per-episode DOF is in qpos."""
    rngs = jax.random.split(jax.random.PRNGKey(seed), batch)
    return np.asarray(jax.jit(jax.vmap(env.reset))(rngs).data.qpos)


def render(model, qpos, grayscale=True, use_textures=True, use_shadows=False,
           res=64, camera="egocentric-rodent", groups=(0, 1, 2)):
    """The policy's own pixels, ``(batch, res, res, C)`` float32 in [0, 1].

    ``mjx.make_data`` is given the env's ``naconmax`` / ``njmax``: the library
    defaults overflow this model's broadphase and the resulting truncated
    contact set moves a 30-step rollout by 5.5e-02, three orders of magnitude
    above the backend's own noise floor.
    """
    mx = mjx.put_model(model, impl="warp")
    d0 = mjx.make_data(model, impl="warp", naconmax=20 * 1024, njmax=400)
    data = jax.jit(jax.vmap(lambda q: mjx.forward(mx, d0.replace(qpos=q))))(
        jnp.asarray(qpos))
    renderer = JaxVisionRenderer(
        mj_model=model, mjx_model=mx, nworld=qpos.shape[0], width=res, height=res,
        grayscale=grayscale, use_textures=use_textures, use_shadows=use_shadows,
        camera_name=camera, enabled_geom_groups=list(groups))
    return np.asarray(jax.block_until_ready(jax.jit(renderer.render)(data)))


def segment(model, qpos, skin_names=(), **kw):
    """Per-pixel wall / floor / treat masks, from a flat-shaded hue render.

    A deep copy of the compiled model gets ``geom_matid = -1`` and pure-hue
    ``geom_rgba``; the ambient term is near-neutral so the hue survives and the
    dominant channel identifies the surface.  Cheaper and exact where a
    threshold on the real image would be circular.
    """
    m = copy.deepcopy(model)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
             for i in range(m.ngeom)]
    skins = set(skin_names)
    wall = [i for i, n in enumerate(names)
            if n in skins or (not skins and n.startswith("maze_wall_"))]
    floor = [i for i, n in enumerate(names) if n == "floor"]
    treat = [i for i, n in enumerate(names) if n.startswith("treat_")]
    rest = [i for i in range(m.ngeom) if i not in set(wall + floor + treat)]
    m.geom_matid[:] = -1
    m.geom_rgba[wall] = [1, 0, 0, 1]
    m.geom_rgba[floor] = [0, 0, 1, 1]
    m.geom_rgba[treat] = [1, 1, 0, 1]
    m.geom_rgba[rest] = [0.25, 0.25, 0.25, 1]
    img = render(m, qpos, grayscale=False, use_textures=False, **kw)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lit = img.max(-1) > 0.004
    return {
        "wall": lit & (r > 0.02) & (g < 0.4 * r) & (b < 0.4 * r),
        "floor": lit & (b > 0.02) & (r < 0.4 * b) & (g < 0.4 * b),
        "treat": lit & (np.abs(r - g) < 0.02) & (r > 2 * b) & (r > 0.02),
    }


def horizontal_coherence(px, mask):
    """Lag-1 horizontal correlation over pixel pairs that are both on ``mask``.

    The discriminator ``std`` cannot be.  A two-tone checker has the same
    within-wall std at every tile size (measured 0.247 at 0.05 / 0.10 / 0.20 m)
    because std only reports the texture's own contrast; what changes with tile
    size is whether that contrast lands as *structure* or as per-pixel aliasing.
    ``r1 -> 1`` means smooth, resolved edges; ``r1 -> 0`` means the tile has
    fallen below the pixel grid and the wall is now noise.
    """
    a, b = px[:, :, :-1], px[:, :, 1:]
    pair = mask[:, :, :-1] & mask[:, :, 1:]
    if pair.sum() < 8:
        return float("nan"), float("nan")
    x, y = a[pair], b[pair]
    return float(np.corrcoef(x, y)[0, 1]), float(np.mean(np.abs(x - y) > 0.08))


def measure(image, masks, tag):
    """One line of brightness + structure statistics; returns the dict printed."""
    px = image[..., 0]
    out = {"tag": tag, "frame_mean": float(px.mean()),
           "frac_dark": float(np.mean(px < 0.15)),
           "frac_clipped": float(np.mean(px > 0.995))}
    for key, mask in masks.items():
        vals = px[mask]
        out[f"{key}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{key}_std"] = float(vals.std()) if vals.size else float("nan")
    out["wall_r1"], out["wall_edgefrac"] = horizontal_coherence(px, masks["wall"])
    print(f"{tag:42s} wall {out['wall_mean']:.3f}+-{out['wall_std']:.3f} "
          f"r1 {out['wall_r1']:+.3f} edge {out['wall_edgefrac']:.3f}  "
          f"floor {out['floor_mean']:.3f}+-{out['floor_std']:.3f}  "
          f"treat {out['treat_mean']:.3f}  frame {out['frame_mean']:.3f}  "
          f"dark {out['frac_dark']:.3f}  clip {out['frac_clipped']:.3f}")
    return out


def save_strip(image, path, scale=4):
    """Nearest-neighbour upscaled filmstrip of a grayscale batch."""
    a = (np.clip(image[..., 0], 0, 1) * 255).astype(np.uint8)
    a = np.repeat(np.repeat(a, scale, axis=1), scale, axis=2)
    imageio.imwrite(path, np.concatenate(list(a), axis=1))


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def variant_specs(env, floor_texrepeat=8, tile=0.10, tilt=1.0):
    """``{name: (spec_or_None, skin_names)}``; ``None`` means "as compiled"."""
    out = {"A_current": (None, ())}

    spec = env._spec.copy()
    spec.material("maze_wall_mat").textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = ""
    spec.material("maze_wall_mat").rgba = [1, 1, 1, 1]
    spec.material("grid").texrepeat = [floor_texrepeat, floor_texrepeat]
    set_four_directional_lights(spec, tilt)
    out["B_box_notex_4lights"] = (spec, ())

    spec = env._spec.copy()
    skins = add_wall_skins(spec, tile=tile)
    spec.material("grid").texrepeat = [floor_texrepeat, floor_texrepeat]
    set_four_directional_lights(spec, tilt)
    out["C_skin_checker"] = (spec, tuple(skins))

    try:
        from labmaze import assets as labmaze_assets
        path = labmaze_assets.get_wall_texture_paths("style_01")["yellow"]
    except Exception:  # labmaze is a dm_control dep; skip if it moved
        return out
    spec = env._spec.copy()
    skins = add_wall_skins(spec, tile=0.30, texture_file=path)
    spec.material("grid").texrepeat = [floor_texrepeat, floor_texrepeat]
    set_four_directional_lights(spec, tilt)
    out["D_skin_labmaze"] = (spec, tuple(skins))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", default="variants",
                        choices=["variants", "tile", "tilt", "all"])
    parser.add_argument("--out", default=None, help="write filmstrip PNGs here")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--res", type=int, default=64)
    args = parser.parse_args()

    env = MazeForageVision(config=default_config())
    qpos = reset_qpos(env, args.batch, args.seed)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    print(f"maze {env.maze_grid.shape} corridor {env.corridor_width:.4f} m, "
          f"{env.n_treats} treats, vision {env.vision_shape}, batch {args.batch}")

    if args.what in ("variants", "all"):
        for name, (spec, skins) in variant_specs(env).items():
            model = env.mj_model if spec is None else spec.compile()
            image = render(model, qpos, res=args.res)
            measure(image, segment(model, qpos, skins, res=args.res), name)
            if out:
                save_strip(image, out / f"ego_{name}.png")

    if args.what in ("tile", "all"):
        print("-- wall texture tile size (checker, 4 lights tilt 1.0) --")
        for tile in (0.05, 0.10, 0.20, 0.40):
            spec = env._spec.copy()
            skins = add_wall_skins(spec, tile=tile)
            spec.material("grid").texrepeat = [8, 8]
            set_four_directional_lights(spec, 1.0)
            model = spec.compile()
            image = render(model, qpos, res=args.res)
            measure(image, segment(model, qpos, tuple(skins), res=args.res),
                    f"tile={tile:.2f} m")

    if args.what in ("tilt", "all"):
        print("-- light tilt (checker tile 0.10) --")
        for tilt in (0.35, 0.6, 1.0, 1.5, 2.0):
            spec = env._spec.copy()
            skins = add_wall_skins(spec, tile=0.10)
            spec.material("grid").texrepeat = [8, 8]
            set_four_directional_lights(spec, tilt)
            model = spec.compile()
            image = render(model, qpos, res=args.res)
            measure(image, segment(model, qpos, tuple(skins), res=args.res),
                    f"tilt={tilt:.2f}")

    if out:
        print(f"wrote filmstrips to {out}")


if __name__ == "__main__":
    main()
