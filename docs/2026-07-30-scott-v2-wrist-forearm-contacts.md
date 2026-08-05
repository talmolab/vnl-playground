# scott_v2 — wrist and forearm contact geoms: handoff (2026-07-30)

Handoff for a collaborator picking up **scott_v2**, the contact-geometry
iteration on the mouse arm+hand+joystick imitation task. Read
`2026-07-28-v25-joystick-contact-handoff.md` first if you have not worked on
this rig — it explains what v25 is and how to run training. This document
covers only what changed in v1 → v2 and what is worth doing next.

> **The Janelia bone meshes are not in this repo and must not be committed.**
> The XMLs reference them by filename only (106 `<mesh file="...">` entries, no
> embedded vertex data). See [Getting the meshes](#getting-the-meshes).

## What scott_v2 is

scott_v1 gave the hand 19 contact-enabled bones — every metacarpal and phalanx
as a per-bone minimum-volume primitive — and nothing above the metacarpals.
scott_v2 adds **three** contact geoms: one block over the seven welded carpals,
and capsules on the distal 40% of the radius and the ulna.

**Exactly one thing changes from scott_v1: the set of contacting geoms.** Every
reward scale, weight, contact parameter, timestep and the whole kinematic tree
are inherited untouched, and the added geoms are `density="0"` so body mass and
inertia come out bit-identical on all 51 bodies (asserted at emit time). v1 vs
v2 is a clean single-factor comparison, which v1 vs v25 deliberately was not.

## Paths you need

| What | Path |
|---|---|
| Branch | **`scott_claude/scott-v2-wrist-forearm-contacts`** |
| v2 walker XML | `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_scott_v2_wrist_forearm_arm_hand_joystick.xml` |
| v1 walker XML (the template) | `.../mouse_forelimb_right_janelia_scott_v1_mixed_arm_hand_joystick.xml` |
| Env + configs | `vnl_playground/tasks/mouse/imitation_arm_hand.py` — `MouseImitationArmHandScottV1`, `default_config_scott_v1()`, `default_config_scott_v2()` |
| Contact presets | `vnl_playground/tasks/mouse/contact_presets.py` |
| XML path resolution | `vnl_playground/tasks/mouse/consts.py` — `janelia_scott_v2_xml_path()`, `janelia_mesh_dir()` |
| Training entrypoint | `vnl_playground/train_mouse_janelia_arm_hand.py` (`--scott-v2`) |
| Bone meshes | **not in the repo** — see below |

`MouseImitationArmHandScottV1` is reused unchanged for v2. It selects grip geoms
by `geom_contype == GRIP_CONTYPE` and enumerates contact-pair rows from MJX's
own static pair list, so the three added geoms are picked up with no code change.

### Getting the meshes

The Janelia model's ~113 bone `.obj` files are an unreleased asset. They are
gitignored by construction (`.gitignore`, `vnl_playground/tasks/mouse/xmls/assets/`)
rather than by everyone remembering. `consts.janelia_mesh_dir()` supplies the
path at load time, in this order:

1. `$JANELIA_MODEL_DIR` — explicit override, meshes kept anywhere
2. `vnl_playground/tasks/mouse/xmls/assets/janelia_model_v24/` — the gitignored
   in-repo drop-in
3. `None` — the XML's own `meshdir` is left untouched

Ask Eric (or whoever handed you this) for the `janelia_model_v24` drop, put it
at either location, and every XML in `xmls/` compiles. Nothing else is needed.

## The three geoms

Fitted by the same minimum-volume rule as v1's 19, every one tangent to its bone:

| geom | type | body | size (mm) | volume | fill |
|---|---|---|---|---|---|
| `wrist_block_v2col` | ellipsoid | `N_L_C_right` | 1.177 × 1.261 × 0.832 | 5.17 mm³ | 41.8% |
| `radius_distal_v2col` | capsule | `radius_right` | r 0.638, h 1.847 | 5.82 mm³ | 48.7% |
| `ulna_distal_v2col` | capsule | `ulna_right` | r 0.485, h 2.482 | 4.15 mm³ | 47.9% |

All three carry the same contact block:

```xml
contype="4" conaffinity="8" condim="4" group="1"
friction="4 0.05 0.05" solref="0.002 1" solimp="0.95 0.95 0.0001 0.5 2"
density="0"
```

Three properties are load-bearing and are asserted, not assumed:

- **`density="0"`.** The bone meshes are all `density="0"`, so the existing
  `*_col` geoms are the *only* source of each segment's mass — the humerus
  capsule's volume × 1000 kg/m³ reproduces its 370.029 mg exactly. A new geom at
  default density would silently add mass to the forearm and v2 would stop being
  an A/B against v1's dynamics.
- **`contype=4` / `conaffinity=8`.** Not decoration:
  `contact_presets.apply_contact_preset` selects geoms by
  `geom_contype == GRIP_CONTYPE`, and `MouseImitationArmHandScottV1` builds its
  contact-pair row index the same way. Added geoms are therefore hardened and
  rewarded automatically.
- **New geoms are additions, never edits.** The seven carpal `*_col` ellipsoids
  and the two forearm capsules stay contact-disabled and keep their sizes,
  because they are what carry the mass. The v2 geom sits alongside them.

Two sizing choices worth knowing before you re-fit anything:

**One wrist proxy, not seven.** Metacarpals 2–5 and the seven carpals carry no
joints, so MuJoCo welds them all to `N_L_C_right` (`body_weldid` of each equals
the wrist's, `body_dofadr = -1`). Measured vertex drift of a wrist-rigid proxy
over the full joint range is 1e-14 m — exact, not an approximation. Seven
separate carpal ellipsoids total 2.43 mm³ but cost 21 contact pairs; one block
is 5.17 mm³ in 3, and the extra volume is mostly in the gaps *between* carpals,
which no real wrist has either.

**Distal 40%, not the whole bone.** The ulna's radius is set by the olecranon at
the elbow, so a whole-bone capsule inflates the shaft to a radius it does not
have (r 0.910 vs 0.485). The distal slice is 8.7× smaller by volume and covers
the only part that can reach the stick.

## The contact approximation, in three layers

These are independently tunable and it is worth not conflating them.

### 1. Geometry: bone mesh → minimum-volume primitive

The smallest ellipsoid or capsule containing *every* bone vertex, oriented by
the vertex PCA frame, with the centre **solved for** rather than fixed at the
centroid. Written as `{x : ‖diag(d)·x + b‖ ≤ 1}` the ellipsoid problem is convex
in `(d, b)`, so this is the global optimum for that orientation. Capsules get 4
free parameters (axis-segment position and half-length) along the first
principal direction; unlike ellipsoids they are analytic in MJX rather than
going through the iterative SDF collider.

Solving for the centre matters because a bone's proximal base and distal head
flare by different amounts, so the centroid sits off the best placement and the
proxy has to balloon on one end to reach the other.

### 2. Physics: `apply_contact_preset`

This is the layer that matters most — at the XML's shipped parameters a 19-geom
anatomical hand buys literally nothing.

| preset | what it does |
|---|---|
| `shipped` | no-op |
| `hard` | `gap=0` + `solimp` dmax on every grip/joystick geom. Leaves `solref` alone. |
| `harder` | `hard` **plus** a direct negative `solref` at `stiffness_mult` × the shipped k, applied to the grip/joystick geoms *and* to the joystick's own slide limits. |

Two non-obvious points:

- **`hard` and `harder` are different mechanisms, not a stiffness ladder.**
  `apply_contact_preset` reads `stiffness_mult` *only* in the `harder` branch, so
  setting `contact_preset: "hard"` alongside `contact_stiffness_mult: 30.0`
  leaves the multiplier inert. On the policy-free probe `hard` measured **0.76
  mN/mm** against a return spring needing ~54 mN — i.e. ~71 mm of
  interpenetration on a hand whose entire contact budget is 1–3 mm.
- **The negative `solref` is the only way past MuJoCo's REFSAFE clamp.**
  `timeconst` is floored at `2·dt`, and the XML ships 0.002 s — sitting exactly
  on the floor. `solref[0] <= 0` switches MuJoCo to `k = -solref[0]/dmax²`,
  bypassing it.
- The joystick's own `x_slide`/`y_slide` limits are hardened too, because they
  inherit the *global* solref (the XML never sets `jnt_solref`). Skip that and
  they become the weakest link and the stick gets squeezed through its own
  ±6 mm range.

`k=30×` at `sim_dt=0.25 ms` is the probe's measured sweet spot: 62 mN of push
against the ~54 mN the spring needs, with 10 mN of ringing. `k=3000×` reaches a
nominally better transmission (0.964 vs 0.920) but delivers 1036 mN oscillating
at ±687 mN, and halving the timestep does not fix it (687 → 674), so it is
genuine stiff-contact ringing rather than discretisation.

### 3. Reward: exact clearance, not a radius proxy

`MouseImitationArmHandScottV1._joystick_contact_reward`. v25 built
surface-to-surface clearance as `center_distance − (r_grip + r_joystick)`,
reading each radius as `geom_size[id][0]`. That is only correct when every geom
is a sphere; scott_v1's proxies are 6 ellipsoids and 13 capsules, where
`geom_size[0]` is the *longest* semi-axis (3–5× the other two) or a capsule
radius with its length ignored. Both sides are replaced with exact geometry:

- **Sim side** — `data.contact.dist`, MJX's own signed distance, exact for any
  geom type. MJX's jax pipeline enumerates a *static* pair list, so the relevant
  rows are located once at construction. Cross-checked against `mj_geomDistance`
  over 12 random poses × 57 pairs: agree to 0.052 mm max (0.001 mm median) under
  2 mm separation, diverging only beyond 5 mm where the reward is already flat.
- **Reference side** — precomputed once at construction with `mj_geomDistance`.
  It depends only on the clip data, so there is no reason to approximate it at
  all, let alone every step.

The sim clearance is clamped at 0. v25's term was an *even* function of
clearance — pressing in scored the same as hovering the same distance away — so
over the 519M-step v25 run the grip deepened and this term's reward *fell*. It
was penalising the policy for gripping harder.

## What is measured, and what is not

**The motivating finding (v1).** The wrist and forearm collision proxies already
exist in the v1 XML with `contype=0 conaffinity=0`, and exact `mj_geomDistance`
says the trained policy lives inside them: the carpal block on **46.9%** of
rollout frames (deepest 1.73 mm), the distal ulna on 26.7%, the distal radius on
9.2%. Specifically the wrist is inside the 2 mm **ball** on 47.1% of all frames
and inside the stem on only 7.7% — the hand wraps the ball so tightly that the
carpal block occupies it. That grasp is not physically realisable, and v2
forbids it.

**A causal claim that was made and is REFUTED.** An earlier version of this work
said the pass-through is *why* the policy pushes the stem. That was asserted
without a test. On a5's own trajectory, P(a v2 geom is inside the joystick |
stem contact carrying force) = **0.086** against **0.742** given ball contact
only — 0.12×, where >2× was predicted. When the hand reaches down to the stem
the wrist is 1.46 mm *clear*. Stem-pushing and the pass-through are separate
phenomena.

**H4, registered before the numbers existed and confirmed.** The three new geoms
are load-bearing rather than acting only as barriers: **32.7%** of all contact
impulse over b1's own rollout, all of it on the ball, none on the stem. The
wrist block is universal (11.6–42.4%, every clip); the forearm is situational
(2 clips substantially, 3 at all). This is deliberately a *within-run*
measurement — see the caveat below.

**Stem-pushing halved, but not attributable to the geometry.** a5 (v1) delivered
21.9% of impulse to the stem; b1 (v2) 11.2%. **b1 differs from a5 in six
factors** — contact set, contact preset, solver iterations, `njmax`, `naconmax`,
and the checkpoint step compared (199.2M vs 293.6M). The hypothesis is that the
wrist/forearm geoms give the policy a proximal push surface on the ball so it no
longer reaches past to the stem; the experiment that would test it is
`scottv1-matched-c0`, scott_v1 at b1's exact preset, solver settings and
buffers, compared at matched steps. **That has not been run.** Until it does,
the halving is not evidence for the geometry.

## Throughput

RTX 5090, 4096 envs, jax backend. v2 costs ~8%:

| variant | geoms | pairs | SDF pairs | env steps/s | rel |
|---|---|---|---|---|---|
| scott_v1 mixed | 19 | 57 | 18 | 3,861 | 1.00 |
| **+ wrist + forearm (v2)** | **22** | **66** | **21** | **3,560** | **0.92** |
| + wrist + forearm + palm | 23 | 69 | 24 | 3,450 | 0.89 |

At the actual training configuration the cost is not measurable: a 4M-step v2
smoke reached `training/sps` **8,909** against 8,829 for the recommended v1
config, with 0 overflow events and `nan_termination` 0.

**Benchmark one variant per process.** Measured back to back in one interpreter,
throughput tracks *measurement order* rather than the variant (`v1_mixed` gave
4,116 measured first and 2,404 measured third). Each env holds its own compiled
programs and device buffers. The first two processes of a cold-card sweep are
also 2.2–2.5× low and must be discarded.

## Buffer sizing — two bugs inherited from a5

| | a5 ran | measured peak | v2 uses |
|---|---|---|---|
| `njmax` (constraints per world) | 256 | **369** under v1, **423** under v2 | 512 |
| `--naconmax-per-world` | 10 | **16** under the trained policy | 20 |

The `naconmax` one is the interesting failure: the original sizing sweep used
**random actions**, and under random actions the arm flails off the stick — peak
simultaneous contacts is **1**. Only the *trained* policy grips: more than 10
contacts are live on 16.1% of frames, peaking at 16. A sizing sweep driven by
random actions cannot see a buffer that only a competent policy fills.
Independently, a5 emitted 4,056 device-side `broadphase overflow` events over 13
hours — and fixing that is **1.46× faster**, because the overflow path itself is
what costs.

Note that Warp treats `naconmax` as a batch total rather than a per-world
budget. Re-measure before trusting headroom under a different backend.

## Running it

```bash
python -m vnl_playground.train_mouse_janelia_arm_hand --scott-v2 \
    --mujoco-impl warp --sim-dt 0.00025 \
    --iterations 8 --ls-iterations 8 \
    --drop-reward joystick_contact --num-envs 2048 \
    --naconmax-per-world 20 --njmax 512 \
    --reward-override joints.weight=6.0 \
    --reward-override joints.exp_scale=1.5 \
    --num-timesteps 300000000 --eval-every 10000000
```

- **No `--contact-preset`.** The env default is already `harder` at
  `stiffness_mult=30`, which is the setting that can actually drive the stick.
  Passing `hard` (as a5 did) silently switches to a different mechanism and
  makes the multiplier inert.
- **`joystick_contact` is dropped under Warp** because Warp exposes contacts as
  one flat batch-wide array keyed by `contact__worldid` rather than a per-world
  `contact.dist`, which the reward's row-index approach cannot read. Under the
  jax backend the term works.
- `sim_dt=0.25 ms` and `num_envs=2048` are not free choices: every `harder`
  system goes unstable at 0.5 ms and above, and 4096 envs OOMs on a 5090.

## Uncommitted variants

Only `wrist_forearm` is in the repo. The others were fitted and measured and can
be re-emitted from the analysis directory:

| variant | added geoms | pairs | SDF pairs |
|---|---|---|---|
| `wrist` | `wrist_block` ellipsoid | 60 | 21 |
| `forearm` | radius + ulna distal capsules | 63 | 18 |
| `wrist_forearm` | both of the above *(committed)* | 66 | 21 |
| `wrist_forearm_full` | whole-bone forearm capsules | 66 | 21 |
| `wrist_forearm_palm` | + `palm_pad` ellipsoid | 69 | 24 |

`janelia_scott_v2_xml_path()` raises on anything but `wrist_forearm` rather than
returning a path that does not exist.

The fits for **all** candidates — including the unadopted `palm_pad`,
`palm_block`, whole-bone forearm capsules and the seven per-carpal ellipsoids —
are in `docs/scott_v2_proxies.json` (size, pos, quat in the parent body's frame,
plus volume, convex-hull fill and the containment check). That is enough to emit
any variant without the analysis directory. The three adopted geoms' parameters
are also directly readable in the committed XML.

## Where to iterate

Ranked by what would actually settle something:

1. **Run `scottv1-matched-c0`** — scott_v1 at b1's exact preset, solver
   iterations and buffers. Without it, nothing about b1-vs-a5 is attributable to
   the geometry.
2. **Re-measure H4 at the final checkpoint.** The 32.72% impulse-share figure is
   at 199.2M; b1 ran to 304M and the largest force in the H4 table (402.49 mN on
   the radius capsule) is 233.49 mN there. Peak force and integrated impulse are
   different quantities and the contact fraction went *up*, so the direction is
   genuinely unknown.
3. **The palm.** V1's four metacarpal proxies are still four bone-tight rods
   with gaps between them rather than a surface. `palm_pad` is fitted and
   rendered but not adopted — one factor at a time. Because MC2–5 are welded, a
   single pad could also *replace* all four, dropping 12 pairs to 3.
4. **Soft tissue.** Every fit is bone-tight. `--soft-tissue-um` exists in the
   emitter and defaults to 0. There is still no skin or paw-pad mesh in the
   asset set.
5. **The sesamoids.** All 14 sit palmar to the MCP joints and are the natural
   knuckle contact surface. Still contact-disabled, as in v1.

## Full analysis

The fitting, feasibility, hypothesis-test and benchmark scripts, the figures,
and the rollout videos live outside this repo in
`analysis/2026-07-29-scott-v2-wrist-forearm-contact-geoms/` — ask Scott for it.
`README.md` there is the long form of this document; `scripts/fit_v2_proxies.py`
and `scripts/emit_v2_xml.py` are what produced the committed XML.
