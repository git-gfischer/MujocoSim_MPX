# Gait, duty factor, step frequency, and MPC gains

Gait in this stack is not just `step_freq`. It is three coupled pieces, and MPC `W` is a fourth, separate tuning layer. You do not always need a new `W` for every tiny duty tweak, but you do need a consistent `(timer_t, duty_factor, step_freq)` set per gait, and you usually retune `W` when the support pattern or swing timing changes a lot.

Primary Go2 locomotion knobs live in `mpx/config/robot_config/config_go2.py` (`Go2Locomotion`). The gait clock is `timer_run` in `mpx/utils/simulation_utils/sim_utils.py`. Foot references and contact sequence are built in `mpx/utils/quad_utils_locomotion/reference_generator_locomotion.py`.

## What each knob actually does

Phase `leg_time` lives in `[0, 1)`, advances by `dt * step_freq`, and a foot is in stance while `leg_time < duty_factor`.

### `timer_t` — who is in phase with whom

This is the gait **pattern**. Order is FL, FR, RL, RR.

```python
# mpx/config/robot_config/config_go2.py  (Go2Locomotion)
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # trot
# timer_t = jnp.array([0.5, 0.0, 0.5, 0.0])  # pace
# timer_t = jnp.array([0.25, 0.75, 0.0, 0.5])  # crawl
# timer_t = jnp.array([0.5, 0.5, 0.0, 0.0])  # bound (not reliable)

duty_factor: float = 0.65
step_freq: float = 1.35  # Hz
```

- **Trot** pairs diagonals (FL+RR vs FR+RL).
- **Pace** pairs left/right (ipsilateral).
- **Crawl** spaces the four legs ~90° apart.
- **Bound** pairs front/hind; marked unreliable with the current locomotion `W`.

### `duty_factor` — fraction of the cycle in stance

- Stance time = `duty_factor / step_freq`
- Swing time = `(1 - duty_factor) / step_freq`

Current trot uses **0.65**, so there is overlap (brief 4-foot support) between diagonal switches.

| Duty | Typical meaning |
|---|---|
| `1.0` | Always stance (balance / standing). Timer unused. |
| `~0.75–0.85` | Crawl: keep three feet down. |
| `0.65` | Walking trot with overlap (current Go2 default). |
| `0.5` | Pure walking trot: no overlap, instant diagonal switch. |
| `< 0.5` | Flight phase (running-like / bound). |

For crawl, the phase offsets only produce stable 3-support if duty is high enough. With 90° spacing, duty below ~0.75 will drop to 2-support, which crawl cannot hold.

### `step_freq` — how fast the cycle runs

This is cycle rate in Hz. It does **not** set forward speed. Commanded speed is the MPC velocity input.

The foothold is Raibert-like:

```text
f1 = 0.5 * v * duty_factor / step_freq
```

At fixed `v`, raising `step_freq` shortens stride; lowering it lengthens stride. To trot *faster* you raise commanded `v`, and you often also raise `step_freq` so steps do not become huge and swing does not get too slow.

Live console clamps (Aliengo interactive CLI) are `step_freq ∈ [0.4, 2.0]` and `duty_factor ∈ [0.4, 0.9]`. Those bounds are a practical range, not a hard physics limit.

## Faster trot is not “only `step_freq`”

If you only bump `step_freq` and leave everything else:

- swing gets shorter and more aggressive (same `step_height` in less time)
- stance GRFs must be applied in less time
- footholds shrink, so the same `v` is tracked with choppier steps

Duty usually needs a small co-adjustment. Faster / running-like trot often **lowers** duty (less overlap, more aerial). A conservative fast walk might keep ~0.65. There is no single formula in the repo; `0.65 / 1.35 Hz` is the tuned walking-trot pair.

## Do you need new MPC gains (`W`) every time?

**Not for every duty value, but yes whenever the contact/support problem changes enough that the old cost is the wrong tradeoff.**

`W` is not a gait parameter. It is one locomotion cost matrix. The same `W` is used for whatever contact sequence the timer produces. GRF *references* auto-scale with the number of stance feet (`mg / n_stance`); the **weights** do not.

| Block | What it punishes |
|---|---|
| `Qp` / `Qrot` | base position / orientation error |
| `Qdp` / `Qomega` | linear / angular velocity error |
| `Qleg` | foot tracking (very strong in Z) |
| `Q_grf` | GRF vs the reference `mg / n_stance` |
| `Qq`, `Qdq`, `Qtau` | joints and torques |

Current locomotion weights (`Go2Locomotion.W`):

```python
Qp     = diag([0, 0, 1e4])           # xy free, z stiff
Qrot   = diag([1000, 1000, 0])       # roll/pitch, yaw free
Qq     = 1e-1 * I_joints
Qdp    = 5e3 * I_3
Qomega = 1e2 * I_3
Qdq    = 1e-1 * I_joints
Qtau   = 1e-1 * I_joints
Q_grf  = 1e-2 * I_12
Qleg   = tile([1e4, 1e4, 1e5])       # per foot xyz
```

This repo already has two different `W`s: locomotion vs balance. Balance is stiffer on pose (`Qp` xy is `9e2` instead of `0`; `Qrot` yaw is `2200` instead of `0`) because the robot is not supposed to walk. Bound is marked unreliable — same hint that pattern + duty + `W` have to match.

### Practical rule

1. **Same gait, small change** (e.g. trot `1.35 → 1.5` Hz, duty `0.65 → 0.60`): try keeping `W`. If feet slap or the base bobs, retune `Qleg` and `Qdp` first.
2. **Same gait, large change** (much faster, duty toward `0.5` or below): you usually need `W`. Shorter stance → GRF/impulse problem; faster swing → foot tracking vs smoothness.
3. **Different gait** (trot ↔ crawl ↔ pace): always set `timer_t` **and** `duty_factor` together. Retune `W` (especially `Qrot`, `Qdp`, `Qleg`, `Q_grf`). Crawl is 3-support / 1-swing; trot is 2-support diagonals. The same weights will not sit at the same tradeoff.

You do **not** need a unique `W` for every duty number if two setups share the same support pattern and similar swing timing. You **do** need a consistent `(timer_t, duty_factor, step_freq)` per gait, and a `W` that was tuned for that family.

## What to put in a per-gait config

Treat each gait as one preset:

| Field | Role | Required? |
|---|---|---|
| `timer_t` | pattern | yes |
| `duty_factor` | stance ratio that makes that pattern stable | yes |
| `step_freq` | cycle rate for the speeds you command | yes |
| `step_height` | often too (faster/rougher → higher) | usually |
| `W` | cost tradeoff | shared inside a family if it still tracks; **separate** across families |

There is currently only one locomotion `W` (`Go2Locomotion`). Switching gait in comments only changes `timer_t`; duty and `W` stay at the trot values, which is why crawl/bound will not just work.

Suggested starting points (tune from here, do not treat as final):

- **Walk trot (current):** diagonal `timer_t`, duty `0.65`, `step_freq` `1.35` Hz, existing locomotion `W`.
- **Faster trot:** keep diagonal `timer_t`, duty `≈ 0.55–0.60`, higher `step_freq`, often higher `Qdp` / slightly softer `Qleg` so the QP can finish the swing.
- **Crawl:** crawl `timer_t`, duty `≈ 0.8`, lower `step_freq` (`≈ 0.8–1.0` Hz), and a `W` that is less aggressive on swing tracking and more on roll/pitch.
- **Balance:** duty `1.0`, unused timer, the existing `Go2Balance.W`.
