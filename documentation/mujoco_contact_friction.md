# MuJoCo contact and friction (Go2)

This note is about the **physics plant**, not the MPC friction cone. MuJoCo decides how much a foot can push into the ground and how much it can resist sliding. The controller only *asks* for forces; the solver may or may not be able to realize them.

Related: the MPC's *soft* friction barrier is in [quadruped_whole_body_mpc.md](quadruped_whole_body_mpc.md) §7.1. Reset-time sampling of foot μ and `solref` lives in `mpx/config/sim_config/config_reset_randomization.py`.

Go2 sources used below:

- Feet: `mpx/data/go2/go2_mjx.xml` (`class="foot"`)
- Flat floor: `mpx/data/go2/scene_flat.xml`
- Solver options: `<option cone="pyramidal" impratio="100" …>` on the Go2 model


## 1. Two different “friction” numbers

| Who | What it is | Go2 default |
| --- | --- | --- |
| **MuJoCo plant** | Coulomb μ on geoms. Caps the *real* tangential force the contact solver can produce. | Feet `1.2`, flat floor `2.0` |
| **MPC cost** | Soft cone barrier in `objectives.py` (`μ ≈ 0.5` or `0.7`). Encourages planned GRFs to stay inside a cone. **Not a hard constraint.** | `μ = 0.5` / `0.7` |

If plant μ is **below** the MPC cone, the optimizer can still request a large sideways GRF. The simulator then slips. That mismatch is useful for dataset diversity (contact, GRF, IMU) and is exactly what foot-friction randomization is for.


## 2. What a contact is in MuJoCo

Each simulation step:

1. Collision detection finds geom pairs (foot sphere vs floor plane).
2. Each pair becomes a **contact** in `mjData.contact` with a normal, a penetration/separation `dist`, and a friction frame (normal + two tangents).
3. The constraint solver (Newton, `iterations=50` on Go2) computes contact impulses so the bodies do not keep sinking and so tangential motion respects the friction cone *as well as the solver can*.

`condim` is how many directions that contact is allowed to resist:

| `condim` | Forces | Typical use |
| --- | --- | --- |
| 1 | Normal only | Frictionless |
| 3 | Normal + 2 sliding tangents | Default Go2 *class* geoms (`friction="0.6"`) |
| 6 | Sliding + torsion + rolling | **Go2 feet** (`condim="6"`) |

Feet use `condim=6` and a 3-number friction vector (see §4). The extra spin/roll terms matter for a small spherical foot that would otherwise spin in place.

Collision **margin** on Go2 is `0.001` m. Contacts can exist with a tiny positive `dist`; the collector's `estimate_contacts` therefore also uses a height fallback, not only the solver flag.


## 3. Combining two geoms: `min(foot, floor)`

A contact always involves **two** geoms. MuJoCo takes the **element-wise minimum** of their friction vectors:

\[
\mu_{\mathrm{contact}} = \min(\mu_{\mathrm{foot}},\; \mu_{\mathrm{floor}}).
\]

On the flat scene the floor is `friction="2.0 0.005 0.0001"` and a foot is `friction="1.2 0.02 0.01"`, so sliding μ is

\[
\min(1.2,\; 2.0) = 1.2.
\]

The floor value **2.0 is a ceiling, not the contact μ**. It does not make the ground extra grippy if the foot is slipperier:

| Foot sliding μ | Floor sliding μ | Contact sliding μ |
| --- | --- | --- |
| 0.20 | 2.0 | **0.20** |
| 0.50 | 2.0 | **0.50** |
| 1.00 | 2.0 | **1.00** |
| 1.20 (XML) | 2.0 | **1.20** |
| 3.00 (hypothetical) | 2.0 | **2.00** (floor would cap it) |

Reset randomization writes **only** `model.geom_friction[foot_id, 0]` (sliding). Torsional and rolling stay at `0.02` and `0.01`. The sampled range `0.20 … 1.00` stays below both the XML foot default (1.2) and the floor (2.0), so the logged `friction` **is** the contact μ.

Feet also set `priority="1"`. When two geoms disagree on `solref` / `solimp` / friction mix, the higher-priority geom's contact parameters win. That is why foot XML, not the floor plane, dominates how the foot *feels* besides the `min` on μ.


## 4. The three friction numbers

`friction="slide spin roll"`:

1. **slide** — Coulomb μ in the tangent plane. This is the number people mean by “friction coefficient”. Tangential force is limited by \(\sqrt{f_x^2 + f_y^2} \le \mu f_n\) (elliptic cone) or a pyramidal approximation (Go2).
2. **spin** — torsional friction about the contact normal. Stops a spherical foot from yawing in place.
3. **roll** — rolling resistance. Small on Go2 (`0.01`).

Randomization only changes **slide**. Ice-like patches in `scene_slippery.xml` drop slide to `0.03` on the *terrain* geom instead; `min(1.2, 0.03) = 0.03`.


## 5. How grippy is a given μ?

These are qualitative bands for **this** robot and MPC, not a materials handbook. Analogies are for intuition.

| Sliding μ | Feel in this sim | Rough analogue |
| --- | --- | --- |
| **~1.0–1.2** | Grippy | Dry rubber on concrete. XML feet are 1.2. |
| **~0.7–1.0** | Firm | Slightly dusty / worn rubber. Still above the MPC cone. |
| **~0.4–0.6** | Slightly slippery | Wet rubber. Around MPC `μ ≈ 0.5`, so hard yaw/accel can slip. |
| **~0.2–0.3** | Clearly slippery | Packed snow / grit. Plant cannot hold many of the GRFs MPC asks for. |
| **~0.03** | Ice | `scene_slippery` patch. Not in the reset randomizer range. |

The interesting threshold is the **MPC cone (0.5–0.7)**. Plant μ above that: ground usually holds the planned tangent force. Plant μ below that: the foot slides even when the optimizer thinks it is “inside the cone”.


## 6. Soft constraints: `solref` and `solimp` (not friction)

MuJoCo contacts are **soft**. The solver does not enforce zero penetration with infinite stiffness. Impedance comes from `solref` and `solimp`.

Go2 feet:

```xml
solref="0.02 1"  solimp="0.9 0.99 0.01 0.5 2"
```

### `solref` — time constant and damping

- `solref[0]` = **timeconst** (seconds). Smaller → stiffer, snappier contact. Larger → spongier.
- `solref[1]` = damping ratio. Go2 uses `1` (critically damped). Randomization does not change this.

A useful mental model: the constraint behaves like a mass-spring-damper whose settling time is about `timeconst`. Stiffness scales roughly like \(1 / t^2\), which is why reset sampling of timeconst is **log-uniform** (`0.012 … 0.035` around the XML `0.02`).

This is **compliance**, not slip. A soft contact can still be high-μ (grippy foam). A stiff contact can still be low-μ (hard ice).

### `solimp` — how impedance ramps with penetration

Five numbers: `dmin, dmax, width, midpoint, power`. They interpolate impedance from a small value at first touch to a larger value at deeper penetration. Randomization does not touch `solimp`.

### Solver slip vs Coulomb slip

Even with a large μ, a *soft* frictional constraint can allow a little tangential drift (“solver slip”). Go2 sets `impratio="100"`: frictional constraints are much stiffer than the normal constraint, which reduces that artifact so the μ you set is closer to the μ you feel. `cone="pyramidal"` replaces the round Coulomb disk with a pyramid (linear inequalities, cheaper). Edges of the pyramid can make slip direction slightly axis-aligned compared with an elliptic cone.


## 7. What reset randomization changes

On each `_respawn()`, `ResetRandomizer` can write:

| Knob | Model field | Meaning |
| --- | --- | --- |
| `friction` | `geom_friction[foot, 0]` | Sliding μ, same value on all four feet |
| `solref_timeconst` | `geom_solref[foot, 0]` | Contact softness, same value on all four feet |

It does **not** change floor friction, `solimp`, spin/roll, `impratio`, or the MPC cone `μ`. Those stay as compiled XML / cost code.

Default sampling range for sliding μ: `0.20 … 1.00` (same-or-slipperier than XML 1.2). Config: `mpx/config/sim_config/config_reset_randomization.py`.


## 8. Mental picture

```text
MPC  asks  for  GRF  inside  a  soft  cone  (μ ≈ 0.5–0.7)
                    │
                    ▼
MuJoCo contact solver
  normal:   solref / solimp  (how spongy is the hit)
  tangent:  min(foot μ, floor μ), condim, cone, impratio
                    │
                    ▼
Actual impulse  →  slip if requested tangent > μ_contact * normal
```

To make contacts **softer**, lower `solref` timeconst is the wrong direction — **raise** timeconst. To make contacts **slipperier**, lower foot sliding μ. They are independent axes: ice is hard *and* slippery; foam is soft *and* can still be grippy.
