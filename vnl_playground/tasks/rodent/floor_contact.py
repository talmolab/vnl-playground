"""Exact geometric floor-contact test behind ``MazeForageVision``'s ``contact_termination``.

dm_control's locomotion tasks (``random_goal_maze.RandomGoalMaze``) end an episode
(discount 0) on ANY contact between a walker geom that is not an allowed
ground-contact geom and a ground geom.  The rodent's allowed geoms are every geom
of the bodies ``foot_L``, ``foot_R``, ``hand_L``, ``hand_R`` and ``vertebra_C1``
(``walkers/rodent.py: ground_contact_geoms``; ``find_all('geom')`` is recursive, so
"every geom of a body" includes its descendants: the toes, the fingers and the
whole tail chain).

WHY THIS IS NOT A READ OF ``data.contact``.  Under the warp backend the contact
list is one flat buffer of ``naconmax`` entries shared by every world (the
``DATA_NON_VMAP`` fields, tagged per contact with ``contact__worldid``).  A
termination function runs per world under ``vmap`` and does not know its own world
id, so reading the list means either a segmented reduce over ``naconmax`` entries
for every world or letting ``vmap`` copy the buffers per world -- the
O(``naconmax``) per-world allocation ``nan_termination``'s docstring warns
about.  The floor, however, is a static plane, so "does geom ``g`` touch it"
has a closed form from the geom's own world pose, which every world carries as
``geom_xpos`` / ``geom_xmat``: the lowest point of a convex primitive along world
-z is its centre minus its support extent in that direction.

Contact rule reproduced (``engine_collision_driver.c``): a geom pair is in
``data.contact`` iff it passes the contype / conaffinity filter (or is an explicit
``<pair>``), is not in an ``<exclude>``d body pair, and its signed distance is
below ``margin``, where ``margin`` is the SUM of the two geoms' margins
(see ``_pair_margin``).  Here
``dist`` is the distance from the geom's deepest point to the plane, so the pair
touches iff ``lowest_z - floor_z < margin`` -- exactly the quantity
:func:`clearance` returns, one value per candidate geom.

STALENESS.  After ``mjx.step`` the kinematics (``geom_xpos`` / ``geom_xmat``) are
those of the start of the LAST physics substep, one ``sim_dt`` behind ``qpos``.
That is also the state ``data.contact`` was computed from, so this matches what
dm_control's ``after_step`` reads, and it is the same state the posture
terminations' ``xpos`` reads.

Everything here is host-side planning (numpy, run once at build time) plus one
traceable function; nothing reads the contact buffers.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np

#: dm_control's ``Rodent.ground_contact_geoms`` bodies.
DEFAULT_ALLOWED_BODIES = ("foot_L", "foot_R", "hand_L", "hand_R", "vertebra_C1")

_G = mujoco.mjtGeom
#: Geom types whose lowest point has a closed form, in the order the candidate
#: array is grouped by (so the traced function has no per-geom branching).
_SUPPORTED_TYPES = (
    int(_G.mjGEOM_SPHERE),
    int(_G.mjGEOM_CAPSULE),
    int(_G.mjGEOM_ELLIPSOID),
    int(_G.mjGEOM_BOX),
    int(_G.mjGEOM_CYLINDER),
)


@dataclasses.dataclass(frozen=True)
class ContactPlan:
    """Host-side constants of the floor-contact test (all numpy, none traced).

    Attributes:
        geom_ids: ``(n,)`` int32 model geom ids of the candidate geoms, grouped
            by primitive type (see ``groups``).
        geom_names: ``(n,)`` names of those geoms (diagnostics).
        groups: ``((mjtGeom, start, stop), ...)`` -- contiguous slices of the
            candidate arrays that share a geom type.
        sizes: ``(n, 3)`` float32 ``geom_size`` of the candidates.
        trigger_z: ``(n,)`` float32. World z at or below which candidate ``i``
            touches the floor: ``floor_z + margin + pair_margin``, maximised over
            the floor geoms it can collide with.
        floor_geom_ids: Floor (static plane) geom ids.
        floor_z: World z of the (single) floor plane, or the highest of them.
        margin: The configured extra distance in metres.
        allowed_bodies: The resolved (suffixed) names of the allowed bodies.
        allowed_geom_ids: Every geom of the allowed bodies and their descendants.
        allowed_colliding_geom_ids: The subset of those that could collide with
            the floor under the model's filters (empty for ``vertebra_C1`` in
            ``rodent_no_tail_collisions``).
    """

    geom_ids: np.ndarray
    geom_names: tuple[str, ...]
    groups: tuple[tuple[int, int, int], ...]
    sizes: np.ndarray
    trigger_z: np.ndarray
    floor_geom_ids: np.ndarray
    floor_z: float
    margin: float
    allowed_bodies: tuple[str, ...]
    allowed_geom_ids: np.ndarray
    allowed_colliding_geom_ids: np.ndarray

    @property
    def n_candidates(self) -> int:
        """Number of candidate geoms."""
        return int(self.geom_ids.shape[0])


def geom_z_extent(kind: int, size: Any, xmat: Any, xp: Any = np) -> Any:
    """Support extent of convex primitives along world z (``lowest_z = z - extent``).

    Args:
        kind: One ``mjtGeom`` value shared by every geom in the batch (sphere,
            capsule, ellipsoid, box or cylinder).
        size: ``(n, 3)`` MuJoCo ``geom_size`` (half-extents; ``[r, h, .]`` for a
            capsule or cylinder with ``h`` the half-length along local z).
        xmat: ``(n, 3, 3)`` geom orientation in the world frame
            (``world = xmat @ local``), so ``xmat[:, 2, j]`` is the world-z
            component of local axis ``j``.
        xp: Array module (``numpy`` or ``jax.numpy``).

    Returns:
        ``(n,)`` extent: how far the geom reaches below its centre.

    Raises:
        ValueError: For a geom type with no closed form.
    """
    row = xmat[:, 2, :]  # world-z component of each local axis
    if kind == int(_G.mjGEOM_SPHERE):
        return size[:, 0]
    if kind == int(_G.mjGEOM_CAPSULE):
        return size[:, 1] * xp.abs(row[:, 2]) + size[:, 0]
    if kind == int(_G.mjGEOM_ELLIPSOID):
        return xp.sqrt(xp.sum((size * row) ** 2, axis=-1))
    if kind == int(_G.mjGEOM_BOX):
        return xp.sum(size * xp.abs(row), axis=-1)
    if kind == int(_G.mjGEOM_CYLINDER):
        radial = xp.sqrt(row[:, 0] ** 2 + row[:, 1] ** 2)
        return size[:, 1] * xp.abs(row[:, 2]) + size[:, 0] * radial
    raise ValueError(f"geom_z_extent: unsupported geom type {kind}")


def clearance(
    contact_plan: ContactPlan, geom_xpos: Any, geom_xmat: Any, xp: Any = np
) -> Any:
    """Signed gap of every candidate geom above its floor trigger height.

    ``clearance < 0`` means the candidate touches the floor (penetrates it, or is
    within ``margin + geom margin`` of it).  With ``margin = 0`` and zero geom
    margins this is the distance from the geom's lowest point to the plane, i.e.
    MuJoCo's contact ``dist``.

    Args:
        contact_plan: The build-time constants.
        geom_xpos: ``(ngeom, 3)`` world geom positions (``data.geom_xpos``).
        geom_xmat: ``(ngeom, 3, 3)`` or ``(ngeom, 9)`` world geom orientations.
        xp: Array module (``numpy`` or ``jax.numpy``).

    Returns:
        ``(n_candidates,)`` clearances in metres, in ``contact_plan.geom_ids``
        order.
    """
    ids = contact_plan.geom_ids
    z = geom_xpos[ids, 2]
    rot = xp.reshape(geom_xmat[ids], (-1, 3, 3))
    extents = [
        geom_z_extent(kind, contact_plan.sizes[a:b], rot[a:b], xp)
        for kind, a, b in contact_plan.groups
    ]
    return z - xp.concatenate(extents) - contact_plan.trigger_z


# ---------------------------------------------------------------------------
# Host-side planning
# ---------------------------------------------------------------------------


def _subtree(model: mujoco.MjModel, roots: Sequence[int]) -> np.ndarray:
    """Bool ``(nbody,)``: bodies that are one of ``roots`` or descend from one."""
    in_tree = np.zeros(model.nbody, dtype=bool)
    in_tree[list(roots)] = True
    for body in range(1, model.nbody):  # parents precede children in MuJoCo
        if in_tree[model.body_parentid[body]]:
            in_tree[body] = True
    return in_tree


def _body_pair_excluded(model: mujoco.MjModel, body1: int, body2: int) -> bool:
    signatures = {int(s) for s in model.exclude_signature}
    return ((body1 << 16) + body2) in signatures or (
        (body2 << 16) + body1
    ) in signatures


def _explicit_pair(model: mujoco.MjModel, geom1: int, geom2: int) -> int:
    """Index of the ``<pair>`` joining the two geoms, or -1."""
    for k in range(model.npair):
        pair = (int(model.pair_geom1[k]), int(model.pair_geom2[k]))
        if pair in ((geom1, geom2), (geom2, geom1)):
            return k
    return -1


def _may_collide(model: mujoco.MjModel, geom1: int, geom2: int) -> bool:
    """Whether MuJoCo would generate a contact between the pair if they overlapped."""
    if _explicit_pair(model, geom1, geom2) >= 0:
        return True  # <pair> bypasses the contype / conaffinity filter
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
        return False
    body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
    if model.body_weldid[body1] == model.body_weldid[body2]:
        return False  # both static, or welded together
    if _body_pair_excluded(model, body1, body2):
        return False
    return bool(
        (model.geom_contype[geom1] & model.geom_conaffinity[geom2])
        or (model.geom_contype[geom2] & model.geom_conaffinity[geom1])
    )


def _pair_margin(model: mujoco.MjModel, geom1: int, geom2: int) -> float:
    """The margin MuJoCo applies to the pair (``mj_collideGeomPair``).

    The SUM of the two geoms' margins, regardless of ``priority``, as MuJoCo's
    own contact list shows on a plane and a sphere (margins 0.01 / 0.03 list
    the contact out to dist 0.04, 0.02 / 0.02 out to 0.04, 0.03 / 0.03 out to
    0.06, with the priorities swapped or not; the tests pin this).  An explicit
    ``<pair>`` carries its own margin and ignores the geoms'.  Every margin in
    the shipped walker and floor is 0, so in production this is 0; it is kept
    exact so a model with margins cannot silently shift the threshold.
    """
    k = _explicit_pair(model, geom1, geom2)
    if k >= 0:
        return float(model.pair_margin[k])
    return float(model.geom_margin[geom1] + model.geom_margin[geom2])


def _resolve_body(model: mujoco.MjModel, name: str, suffix: str) -> int:
    """Body id of ``name`` or ``name + suffix``; -1 when neither exists."""
    for candidate in (name + suffix, name):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if body_id >= 0:
            return body_id
    return -1


def plan_contact_termination(
    model: mujoco.MjModel,
    walker_body: str,
    allowed_bodies: Sequence[str] = DEFAULT_ALLOWED_BODIES,
    margin: float = 0.0,
    suffix: str = "",
) -> ContactPlan:
    """Resolves which walker geoms the floor-contact test watches.

    Args:
        model: The compiled CPU model (read only).
        walker_body: Name of the walker's root body (e.g. ``"walker-rodent"``).
        allowed_bodies: Walker body names whose geoms (and their descendants')
            may touch the floor.  Each is looked up as ``name + suffix`` first,
            then as given.
        margin: Extra distance in metres added to every geom's own margin.
        suffix: Name suffix attached when the walker was added to the arena.

    Returns:
        The build-time constants of the test.

    Raises:
        ValueError: If ``margin`` is not finite, the walker body is missing, an
            allowed body matches nothing (or is outside the walker), the model has
            no static floor plane or a tilted one, a candidate geom has a type
            with no closed form (named in the message), or no candidate geom can
            touch the floor.
    """
    if not np.isfinite(margin):
        raise ValueError(f"contact_termination margin must be finite, got {margin!r}.")
    walker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, walker_body)
    if walker_id < 0:
        raise ValueError(f"contact_termination: walker body {walker_body!r} not found.")
    walker_tree = _subtree(model, [walker_id])

    resolved = []
    for name in allowed_bodies:
        body_id = _resolve_body(model, str(name), suffix)
        if body_id < 0 or not walker_tree[body_id]:
            raise ValueError(
                f"contact_termination: allowed body {name!r} matches no body of "
                f"{walker_body!r} (tried {name + suffix!r} and {name!r})."
            )
        resolved.append(body_id)
    allowed_tree = (
        _subtree(model, resolved) if resolved else np.zeros(model.nbody, bool)
    )

    floor_ids = np.array(
        [
            g
            for g in range(model.ngeom)
            if model.geom_type[g] == int(_G.mjGEOM_PLANE)
            and model.body_weldid[model.geom_bodyid[g]] == 0
        ],
        dtype=np.int32,
    )
    if floor_ids.size == 0:
        raise ValueError("contact_termination: the model has no static floor plane.")
    scratch = mujoco.MjData(model)
    mujoco.mj_forward(model, scratch)
    floor_z = {}
    for g in floor_ids:
        normal = scratch.geom_xmat[g].reshape(3, 3)[:, 2]
        if not np.allclose(normal, (0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError(
                f"contact_termination: floor {model.geom(int(g)).name!r} is not "
                f"horizontal (normal {normal}); only +z planes are supported."
            )
        floor_z[int(g)] = float(scratch.geom_xpos[g, 2])

    allowed_geoms = [
        g for g in range(model.ngeom) if allowed_tree[model.geom_bodyid[g]]
    ]
    candidates, triggers = [], []
    for g in range(model.ngeom):
        body = model.geom_bodyid[g]
        if not walker_tree[body] or allowed_tree[body]:
            continue
        thresholds = [
            floor_z[int(f)] + margin + _pair_margin(model, g, int(f))
            for f in floor_ids
            if _may_collide(model, g, int(f))
        ]
        if not thresholds:
            continue  # this geom can never touch the floor
        if int(model.geom_type[g]) not in _SUPPORTED_TYPES:
            raise ValueError(
                f"contact_termination: candidate geom {model.geom(g).name!r} has "
                f"type {_G(model.geom_type[g]).name}; only sphere, capsule, "
                "ellipsoid, box and cylinder have a closed-form lowest point."
            )
        candidates.append(g)
        triggers.append(max(thresholds))
    if not candidates:
        raise ValueError(
            "contact_termination: no walker geom outside the allowed bodies can "
            "collide with the floor, so the criterion could never fire."
        )

    order = sorted(
        range(len(candidates)),
        key=lambda i: (_SUPPORTED_TYPES.index(int(model.geom_type[candidates[i]])), i),
    )
    geom_ids = np.array([candidates[i] for i in order], dtype=np.int32)
    trigger_z = np.array([triggers[i] for i in order], dtype=np.float32)
    types = model.geom_type[geom_ids]
    groups, start = [], 0
    for kind in _SUPPORTED_TYPES:
        count = int((types == kind).sum())
        if count:
            groups.append((kind, start, start + count))
            start += count

    colliding = [
        g
        for g in allowed_geoms
        if any(_may_collide(model, g, int(f)) for f in floor_ids)
    ]
    return ContactPlan(
        geom_ids=geom_ids,
        geom_names=tuple(model.geom(int(g)).name for g in geom_ids),
        groups=tuple(groups),
        sizes=model.geom_size[geom_ids].astype(np.float32),
        trigger_z=trigger_z,
        floor_geom_ids=floor_ids,
        floor_z=max(floor_z.values()),
        margin=float(margin),
        allowed_bodies=tuple(model.body(b).name for b in resolved),
        allowed_geom_ids=np.array(allowed_geoms, dtype=np.int32),
        allowed_colliding_geom_ids=np.array(colliding, dtype=np.int32),
    )
