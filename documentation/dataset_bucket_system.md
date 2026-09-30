# The bucket system: balanced proprioceptive dataset collection

This note is about **how collected timesteps are classified, counted and balanced**, not about how they are recorded. Recording (what columns exist, at what rate, in which parquet file) is in `mpx/utils/dataset_collection/dataset_schema.py` and `episode_recorder.py`. The bucket system is the layer on top: it decides *what kind* of sample each timestep is, keeps the statistics that say whether the dataset is any good, and produces the index a training sampler reads.

Sources:

- `mpx/utils/dataset_collection/dataset_bucket_system.py` — `DatasetBucketSystem`, `BucketKey`, `SampleRef`
- `mpx/utils/dataset_collection/contact_labeling.py` — the **only** definition of contact
- `mpx/utils/dataset_collection/grf_diversity.py` — the post-hoc diversity diagnostics
- `mpx/utils/dataset_collection/dataset_summary_plotter.py` — plot the cross-run JSON
- `mpx/config/sim_config/config_dataset_bucket.py` — every threshold, defined once
- Related: [mujoco_contact_friction.md](mujoco_contact_friction.md) (what a contact is in the plant, and why there is no ground truth to read)


## 1. What a "sample" is

A sample is **one labelled timestep**, not a window.

```
SampleRef = (episode_id, t, + the label statistics needed for balancing)
```

The buckets store no trajectory data at all. Every episode is written once, whole, as `episodes/<episode_id>.parquet`; a bucket holds only lightweight references naming `(episode_id, t)`.

This is deliberate and it is the single most important design choice in the module: **collection is window-length agnostic**. A model that wants a window `W` slices `[t - W + 1 … t]` out of the episode file and skips index rows with `t < W - 1`. One collected run therefore serves *any* `W` — 5, 10, 50 — without recollection, and a window can never straddle an episode boundary because it is always cut from a single file.

Every index row carries `episode_n_steps` alongside `t`, which is what makes that claim true for any `W`. (`window_valid_w10` survives as a convenience column; it hardcodes one `W` and contradicts this section, so do not build on it.)

Stride-1 rows are **not** independent observations. At `label_stride = 1` and `W = 10`, consecutive rows share 9 of their 10 frames, so a variance estimate computed over stride-1 held-out rows is far too tight. `run_metadata.json` records `effective_sample_size` for exactly this reason; use a stride of at least `W` on val and test.


## 2. The bucket key

```python
BucketKey = (contact_state, perturbation_level, speed_bin, terrain, gait_type)
```

All five are read **at that single timestep** `t`. Nothing in the key depends on a window.

| Field | Values | Source |
| --- | --- | --- |
| `contact_state` | 16 named 4-bit patterns (§3) | the `contact` column — the debounced label (§4) |
| `perturbation_level` | `none`, `small`, `large` | `‖F_ext‖` vs a body-weight floor and the sampler's own midpoint (§6) |
| `speed_bin` | `stopped`, `slow`, `medium`, `fast`, `reverse`, `turning` | `cmd_base_vel` (vx, vy, yaw rate) |
| `terrain` | `flat`, `rough`, `stairs` | `TerrainType`, from episode metadata |
| `gait_type` | `trot`, `crawl`, `pace`, `bound`, `balance`, `transition` | `gait_from_phase_offsets(timer_t)` |

The label a bucket key prints (and the value in the `bucket_key` index column) is:

```
DIAG_FL_RR | pert=none | medium | flat | trot
```

### Why speed and perturbation magnitude are in the key

§7 says GRF diversity is enforced upstream by varying **speed, friction, payload and perturbation magnitude**. With none of those in the key, a diagnostic can report that a bucket is homogeneous but never say *which axis collapsed* — and you can have full bucket coverage while every sample in a bucket came from one command.

The audited run demonstrates it: `FULL | perturb=false | flat | trot` held 9,627 samples and looked healthy, and every one of them came from a single constant command per episode.

Perturbation used to be a boolean, which made §7's own advice ("add perturbation magnitude range, not just presence") impossible to act on: a 6 N nudge and a 60 N shove landed in the same bucket. It is now three levels, binned as a **fraction of body weight** rather than in newtons, because payload is randomized 0.5–5.0 kg — the old absolute 5 N threshold was 2.8 % of the Go2's 176 N, inside the noise of the external-force log itself.

**Friction and payload stay out of the key.** They are continuous and would explode it. They are tracked per bucket instead, as `n_randomization_groups` in the diversity report (§7): a bucket fed by one randomization draw is homogeneous however many samples it holds.


## 3. The 16 contact states — and what happened to "24 buckets"

Contact is a 4-bit vector in **FL FR RL RR** order, `1 = stance`, `0 = swing`. All 16 patterns have a name; the 16 patterns and the 16 names are in bijection, so a name identifies a pattern uniquely.

**The 12 nominal gait states** (`NOMINAL_GAIT_STATES`):

| Name | Bits | Meaning |
| --- | --- | --- |
| `FULL` | 1111 | Standing / crawl support / 4-leg balance |
| `SWING_RR` | 1110 | Rear-right swing |
| `SWING_RL` | 1101 | Rear-left swing |
| `SWING_FR` | 1011 | Front-right swing |
| `SWING_FL` | 0111 | Front-left swing |
| `HIND_PAIR` | 0011 | Hind pair stance |
| `FRONT_PAIR` | 1100 | Front pair stance |
| `IPSIL_FL_RL` | 1010 | Ipsilateral left (pace phase) |
| `DIAG_FL_RR` | 1001 | Diagonal FL+RR (trot phase) |
| `DIAG_FR_RL` | 0110 | Diagonal FR+RL (trot phase) |
| `IPSIL_FR_RR` | 0101 | Ipsilateral right (pace phase) |
| `FLIGHT` | 0000 | No foot down (bound flight phase) |

**The 4 single-support states** (`SINGLE_SUPPORT_STATES`): `SINGLE_FL` 1000, `SINGLE_FR` 0100, `SINGLE_RL` 0010, `SINGLE_RR` 0001.

> v3 collapsed all four single-support patterns into one `RARE` bucket, so two different bit patterns shared one label and the label was ambiguous. v4 gives each its own name. `rare_contact` survives as a deprecated boolean column; prefer testing `contact_state.startswith("SINGLE_")`, which also tells you *which* foot.

### Where "24" came from, and why it is no longer the number

Under the **old four-field key**, a single-terrain single-gait run degenerated to `(contact_state, perturbation_active)`:

```
12 nominal contact states × 2 perturbation states = 24 nominal buckets
```

plus 8 more from the four single-support states, giving the **32 active buckets** that `datasets/dataset_summary.json` reports for the flat/trot runs.

Under the **current five-field key**, the same flat/trot run splits further by perturbation level (3) and command regime (up to 6):

```
16 contact states × 3 perturbation levels × 6 speed bins = 288 reachable in one terrain+gait
```

of which only a fraction is physically reachable (no `FLIGHT` while `stopped`, no `DIAG_*` outside a trot phase, and so on). Across the full key space the theoretical maximum is 16 × 3 × 6 × 3 terrains × 6 gaits = 5,184. Neither number is a target: `active_buckets` counts the keys actually hit, and the split is there so that a homogeneous bucket is *visible*, not so that every cell is filled.

The practical consequence is that buckets get smaller, which is why §5 gained a population floor: a bucket with one sample is reported as **not collected** rather than weighted 57× and trained on.

### What a real flat/trot run looks like

Measured over 30 episodes, 58,344 samples (`datasets/dataset_summary.json`, collected under the old key):

| Contact state | Samples | Share |
| --- | --- | --- |
| `DIAG_FR_RL` | 20,574 | 35.3 % |
| `DIAG_FL_RR` | 20,331 | 34.8 % |
| `FULL` | 12,484 | 21.4 % |
| `SWING_RL` | 1,492 | 2.6 % |
| `SWING_RR` | 1,256 | 2.2 % |
| `SWING_FL` | 842 | 1.4 % |
| `SWING_FR` | 787 | 1.3 % |
| `FRONT_PAIR`, `IPSIL_*`, `FLIGHT`, `HIND_PAIR` | 300 | 0.5 % |
| `SINGLE_*` (4 states) | 278 | 0.5 % |

Two trot diagonals plus full support account for **91 %** of a trot dataset. That extreme prior is what the balancing machinery in §5 exists to handle — and exactly why it must not be handled by deleting rows.


## 4. Contact labels: one definition, and it is not in this module

`contact` is produced by `contact_labeling.py`, in three causal stages:

1. **Substep majority.** Per-foot force is accumulated on every sim step; a foot reads "loaded" for the control step when the majority of its substeps exceed `on_threshold_n = 15 N`.
2. **Schmitt trigger.** Touchdown needs 15 N, release needs the force to fall below `off_threshold_n = 5 N`. A force hovering near one rail cannot oscillate.
3. **Minimum dwell.** A state must hold `min_dwell_steps = 3` (60 ms at 50 Hz) before it may change again, which bounds the transition rate whatever the force does.

Every stage uses only current and past samples, so the label is reproducible online and no future information leaks backwards.

### The label lags physics by up to 60 ms, by construction

This is the most important property of the label and the easiest one to mistake for a bug. `min_dwell_steps = 3` at 50 Hz means a foot that has genuinely lifted off keeps reading `contact = 1` for up to three more control steps, and a foot that has genuinely touched down keeps reading `0`. That is what bounds the transition rate; it is not a defect.

The consequence a reviewer will trip over: **zero GRF while `contact == 1` is expected inside that window.** On the audited run 0.1076% of in-contact feet carried exactly 0 N, which looks like the aliasing §5 of the R3 work removed — until you measure where those frames sit:

```
distance from the offending frame to the nearest contact-label transition:
  0 steps  0.0%   1 step 61.2%   2 steps 36.0%   3 steps 2.9%   4+ steps 0.0%
```

Every one is inside the dwell window. Outside it the rate is **0.0%**. `validate_run.py` masks the window before gating, and any analysis that pairs the contact label against a force should do the same — or use `dt_since_transition`, which is exactly the distance being masked on.

**The bucket system never re-derives contact.** It used to carry a second rule — a plain `|GRF| > 5 N` threshold in `derive_contacts`, plus a `contact_force_threshold` in the config. Production never called it (the recorder runs the debouncer), but it was not harmless: it kept a third contact threshold alive in the config, its docstring misdescribed where labels come from, and it labelled the synthetic test fixtures, so those fixtures exercised a path that disagreed with production on **8.6 %** of rows. Measured on the shipped run:

```
index.contact_bits == debounced `contact` column : 100.0000%
index.contact_bits == raw |GRF| > 5 N threshold  :  91.4184%
```

It is gone. There is one definition of contact and it lives in `contact_labeling.py`.

### Why a chosen definition is the honest answer

**MuJoCo has no contact ground truth to give.** `data.contact` lists geom pairs within `margin`, which is a proximity statement, not a load statement; with compliant contact the normal force ramps continuously over the `solref` time constant, so there is no instant at which anything flips. Any binary contact label is a *chosen definition*.

What makes this one defensible is that it is stated, reproducible, causal, and expressible on hardware — and that an independent check agrees with it. `contact_from_height`, built from foot clearance using the convention the real-world self-supervised benchmark uses, agrees with the GRF label on **98.5 %** of frames.

Two supporting choices:

- **Sim rate 500 Hz** (`SimRateConfig.sim_hz`). The contact solver's 10 ms time constant spans 5 substeps at 500 Hz but only 2 at 200 Hz, where the foot visibly bounces.
- **Control / logging rate 50 Hz**, i.e. 10 substeps per control step.

### Plan versus physics: `schedule_mismatch`

`contact_schedule` — what the gait planner said the foot should be doing — is recorded alongside. The disagreement is decomposed, because the raw flag does not isolate the subset it claims to:

```
per-foot mismatch rate: 2.96%        (the .any() rate is 9.47%)

distance from a mismatched frame to the nearest contact transition:
   0 steps 12.6%   1 step 59.1%   2 steps 16.1%   3 steps 6.6%   4 steps 2.8%   >4 steps 2.8%

mismatch run lengths: L=1 3237 runs, L=2 598, L>=3 469
frames in runs of >=3: 2178 of 6611 (32.9%)

best global lead/lag shift: 0 in all 28 episodes
```

**88 % of mismatched frames sit within two control steps of a contact transition, with no systematic lead or lag.** That is touchdown/liftoff timing jitter between planner and physics, not a slip — and a torque-reading model fails on those frames for trivial reasons too.

So the index carries three columns:

| Column | Meaning | Use |
| --- | --- | --- |
| `schedule_mismatch` | any foot disagrees | diagnostic; 88 % of it is jitter |
| `schedule_mismatch_sustained` | inside a run of ≥ 3 mismatched frames | **the benchmark subset** (33 % of mismatched frames) |
| `schedule_mismatch_bits` | which feet disagreed, `"0100"`-style | error analysis |

Report the *sustained* subset. It is the part on which reading the controller's plan genuinely cannot score.


## 4b. Operating regime: how degraded the locomotion is

`post_failure` fired on **52 of 32,306 rows (0.16%)** of the audited run, while crash episodes contributed **3,194 rows (9.9%)**. So 98.4% of the frames inside a crashing episode were labelled as normal operation — including the belly-scraping stumble where the robot is still trying to execute the gait.

The crash predicate was never meant to carry that load. `crash_height_m = 0.15` and `crash_tilt_deg = 60` are *terminate* conditions, and the robot is visibly wrong for more than a second before they trip:

| steps before end | base height | tilt median | tilt p95 |
|---|---|---|---|
| 250–1000 | 0.202 m | 2.9° | 15.9° |
| 120–250 | 0.220 m | 3.4° | 22.2° |
| 60–120 | 0.212 m | 5.6° | 18.7° |
| 30–60 | **0.194 m** | **12.9°** | 24.9° |
| 15–30 | **0.175 m** | 17.3° | 34.8° |
| 5–15 | 0.164 m | 17.7° | 44.4° |
| 0–5 | 0.152 m | 29.6° | 59.4° |

And it is not only crash episodes: episode 00023 of that run terminated `goal_reached` with **76.2% of frames below 0.20 m** — a full minute of walking in a collapsed posture, indistinguishable from a healthy run by anything the dataset recorded.

`operating_regime` grades every frame `nominal` / `degraded` / `severe` / `failed` as the **worst** of four signals:

| Signal | Rails | Note |
| --- | --- | --- |
| `non_foot_contact_n` | > 5 N ⇒ at least `severe` | The most direct of the four and the only one needing no tuning: a healthy quadruped reads exactly 0 N on any geom that is not a foot. Substep-averaged, so a one-substep graze does not register. |
| `base_height_terrain` | 0.195 / 0.175 / 0.150 m | Just below the non-crash p5 and p1. |
| `base_tilt_deg` | 12 / 25 / 45° | Above the non-crash p99 (9.2°) and max (17.3°). One total angle, not roll and pitch separately. |
| `cmd_tracking_error` | > 0.35 m/s ⇒ `degraded` | ‖realised − commanded‖ planar velocity. |

The reference distributions are well separated, which is where the rails come from:

```
non-crash:  height p1 0.183  p5 0.195  median 0.249 | tilt p95  6.4  p99  9.2  max 17.3
crash    :  height p1 0.149  p5 0.165  median 0.208 | tilt p95 24.5  p99 35.3  max 59.7
```

Three rules:

1. **Posture signals are dwell-filtered** (`dwell_steps = 3`, causal, like the contact debouncer) so a noisy frame cannot flip the annotation. **Body contact is not** — it is applied as a floor on top, because a dwell filter must never label a frame `nominal` while the robot's body is carrying load.

   That asymmetry is what makes the chatter check sharp. `apply_dwell` provably cannot emit an interior run shorter than `dwell_steps` (verified over 20k random traces), so the *only* thing that can produce one is the body-contact floor. A sub-dwell run with no body contact in it or on either side is a bug in the filter, not a property of the robot — and the gate asserts exactly that, rather than counting transitions. Counting transitions cannot tell a robot genuinely oscillating around the 0.195 m rail (sustained blocks of 6–40 frames, which the annotation *should* record) from a per-frame flicker; on the audited run one episode flipped 79 times in 1,277 steps and every flip was a real block. Measured there: 360 runs, 22 shorter than the dwell — 3 truncated by the episode boundary, 19 by the body-contact floor, **0 unexplained**.
2. **`failed` is absorbing.** The robot does not get back up, which preserves the v4 `post_failure` latch.
3. **Degraded frames are kept.** Dragging feet, unplanned shin and belly contact, GRFs the gait schedule never intended — that is exactly the regime a contact estimator should be tested on. Report `nominal` and `degraded` accuracy separately rather than pooling or discarding.

`post_failure` is now `operating_regime == "failed"` and `valid` is `operating_regime in (nominal, degraded)`, so both existing columns keep working with sharper definitions. The regime is deliberately **not** a `BucketKey` field — four levels would multiply the key and most combinations are empty — and is tracked as `nominal_fraction_by_bucket()` instead, beside the diversity statistics.

**Every diagnostic in this pipeline is now scoped to nominal frames**, for one reason discovered three separate ways: the clipping audit over all rows was measuring a fallen robot (83.1% of its clipping rows were crash frames, 9.9% of the data); the attitude-estimator RMS was dominated by a handful of short degraded episodes; and the difficulty probe's headroom ratio moved when crash episodes changed folds. In each case the aggregate described the fall rather than the locomotion.


## 5. Balancing: weights, not deletion

**Buckets never drop samples.** `_bucket_add` always stores and always returns `True`.

This was a deliberate reversal. v4.0 equalised the three majority contact states and silently discarded 2,069 of 23,994 rows (11.2 % of `0110`, 10.9 % of `1001`, 6.5 % of `1111`, 0 % of everything else). That changes the class prior: accuracy measured over such an index is not accuracy over the real distribution, and comparing it against MI-HGNN or ECNN numbers computed on unbalanced data is apples to oranges.

Balancing is a **training-time weight**, computed by `bucket_weights()`:

```python
weight(bucket) = (1 / count(bucket)) ** alpha        # alpha default 0.5
```

normalised so the mean weight over weighted samples is 1.0. `alpha = 0` is the natural distribution, `alpha = 1` fully equalises, `0.5` lifts the rare states without pretending they are as common as a trot diagonal.

Three guards, each fixing a way the previous version contradicted its own docstring:

| Guard | Default | Why |
| --- | --- | --- |
| **Counts from `weighted_splits` only** | `("train",)` | Counting over every split let held-out class statistics leak into the training weights. |
| **`min_bucket_samples_for_weighting`** | 50 | Below it a bucket gets weight 0 and is reported by `undercollected_buckets()` as *not collected*. One sample cannot teach a class, only add gradient variance — the audited run had a bucket with n = 1 carrying **57× the modal weight**. |
| **`max_weight_ratio`** | 10.0 | Clipped after weighting, then renormalised. The shipped index spanned 0.7043 → 57.1266, an **81× ratio**. |

Weights are keyed on the **whole bucket label**, not on `contact_state` alone: the key has five fields and the perturbation and speed axes exist precisely so their coverage can be balanced.

Two rules that follow:

1. **Outside the weighted splits the weight is exactly 1.0.** `index_balanced.parquet` carries a `split` column so a reader can tell a real weight from that default. A reweighted val or test metric describes a distribution that does not exist.
2. **Record `alpha`, `max_weight_ratio` and the floor in the manifest.** `save_dataset` writes all three into `run_metadata.json`.

Rows whose weight is 0.0 stay in `index.parquet` — nothing is deleted — they are simply not drawn by a weighted sampler.

`bucket_capacity = 5,000` is a *diagnostic* reference, not an eviction limit.


## 6. Perturbation balance

| Knob | Default | Meaning |
| --- | --- | --- |
| `perturbation_small_bw_frac` | 0.10 | Detection floor: 17.6 N on a bare Go2 |
| `perturbation_force_range_n` | (10, 50) N | The sampler's own range; its midpoint against the floor is the `small`/`large` boundary, 33.8 N |
| `min_perturbation_ratio` | 0.25 | Target minimum perturbed fraction |
| `min_bucket_samples_for_ratio_check` | 100 | Below this a bucket's ratio is noise |

The floor scales with body weight because payload is randomized. The upper boundary does **not**: a fixed 0.35 × 176 N = 61.6 N sat above the sampler's 50 N maximum, so `large` was structurally empty and the key carried a value that could never occur. Splitting the sampler's own range puts the boundary at 33.8 N against a measured perturbed p50 of 32.9 N, and keeps both bins populated if the magnitude is later retuned.

The point of this axis: external-base-force estimation and GRF estimation are two of the four heads the dataset trains, and neither can be learned from a robot that is never pushed. Splitting every contact state by perturbation level is what guarantees each state is seen under every regime, rather than perturbation clustering into whichever states happened to be active when a push landed.

**Check the worst bucket, not the aggregate.** That is the whole rationale for the axis, and an aggregate hides a state that is never pushed. The audited run's global 20.9 % covered a per-state range of **9.4 % to 66.7 %**. `perturbation_ratio_by_bucket()` reports it, `print_summary` shows the worst five, and `diversity_warnings` returns `worst_perturbation_ratio` with the bucket that owns it.


## 7. GRF diversity — post-hoc, off the parquet

GRF balance is **not** enforced by binning force magnitudes at collection time. Doing so would either drop rows (§5) or distort which contact configurations are represented. It is enforced *upstream*, through episode diversity, and verified *downstream* by `grf_diversity.py`, which reads `index.parquet` after the fact.

Reading the written index rather than the in-memory buckets is the point: **the statistic can be changed without recollecting.** The bucket system only counts.

### The old statistic was measuring the wrong thing

`grf_diversity_report()` computed `std` of total GRF per bucket, and it is gone. Three problems:

- **Total magnitude is nearly invariant by construction.** In support, `Σ‖f_i‖` equals body weight plus payload whatever the distribution across feet: a 90/10 front-rear split and a 50/50 split give the same number. So the diagnostic fired hardest on `FULL` support — exactly where low total variance is physics, not a defect — and was blind to what the GRF head actually has to learn. The doc's own worked example flagged `FULL | perturb=false | flat | trot` at `std = 8.3 N`.
- **`std` is not robust here.** The distribution has `p99/p50 = 1.87` and a 732 N maximum from touchdown transients, so `std` mostly reports impacts.
- **`15.0 N` was absolute** on a robot whose payload is randomized 0.52–4.98 kg.

### What replaced it

`bucket_diversity(run_dir)` returns one row per bucket:

| Field | What it catches |
| --- | --- |
| **`load_share_iqr`** | **The one that matters.** Spread of the largest single-foot share of the total load, in [0.25, 1.0]. Statics pins the total; the *distribution* is what varies and what the GRF head must discriminate. Warn below **0.02**. |
| `total_grf_iqr_bw` | Robust total spread, normalised by body weight so the threshold does not move with the payload draw. Warn below **0.08**. |
| `total_grf_p99_over_median` | Above **2.5**, the bucket's only spread is touchdown impacts. |
| **`n_randomization_groups`** | A bucket of 9,627 samples all drawn from one command and one friction draw is homogeneous however full it looks. Nothing else catches this. Warn below **3**. |
| `tangential_ratio_p95` | Friction utilisation, `max_i ‖f_xy,i‖ / f_z,i` over loaded feet — the slip-relevant axis, measured nowhere else. |
| `perturbation_ratio` | Per-bucket, per §6. |

```
python -m mpx.utils.dataset_collection.grf_diversity datasets/<run>
```

`print_summary()` no longer prints a GRF number at all; it points at this tool.

### What to do about a flagged bucket

The fix is always at the episode level, never at the label level:

- **Widen the velocity command.** `segmented_commands = True` drives collection from a segmented velocity command instead of goal following. Goal following never commands a yaw rate and never reverses — the audited v4 run had commanded yaw identically zero for all 23,994 rows, so yaw-invariance could not be tested at all, and the GRF distribution was correspondingly narrow. A collapsed `speed_bin` in the key now points straight at this knob.
- **Resample the randomization more often.** `randomize_per_episode = True` resamples every knob at each episode boundary. v3 reused one parameter set across up to 12 consecutive episodes, which both narrowed the spread and put near-duplicate episodes in different splits. This is what `n_randomization_groups` measures.
- **Vary friction, payload and terrain.** Foot μ and `solref` sampling lives in `config_reset_randomization.py`; see [mujoco_contact_friction.md](mujoco_contact_friction.md).
- **Add perturbation magnitude range**, not just presence — which the `small`/`large` split now makes visible.

Two population diagnostics sit beside it. `undercollected_buckets()` is the absolute floor (< 50 samples: not collected, weight 0). `underpopulated_buckets()` is relative — below 25 % of an even share of the run — because a fraction of `bucket_capacity` flags every bucket on a 5-episode pilot and none on a 500-episode run.


## 8. Episode-level integrity

Three things are handled per episode rather than per sample, because per-sample handling would leak.

**Splits.** Train/val/test is assigned **per episode** (`assign_split`), so overlapping windows cut from one episode can never straddle a split boundary. The index carries no `split` column: the split comes from `datasets/manifest.json` at load time, joined on `randomization_group_id`, because episodes sharing a randomization draw are near-duplicates and must land in the same split. Defaults: `val_ratio = 0.15`, `test_ratio = 0.10`, `split_seed = 0`.

`make_manifest.py` gives each **stratum its own row quota** rather than sharing one. With a shared quota the greedy largest-first walk sent every crash episode to val: val has the smallest quota so it fills last and with the smallest groups, and crash episodes are short (55–1,277 steps against 1,500–2,100 for completed ones), so "smallest groups" and "crash groups" are the same set. That correlation is structural and does not weaken as the dataset grows. On the audited run it left val **100% crash episodes and 46% nominal rows** against 94% for train and test — model selection tuned on the fallen-robot regime. Per-stratum quotas bring val to 82.6% nominal. The manifest then asserts the *outcome* rather than the procedure: every split's `frac_nominal` must sit within 0.10 of the dataset's, which catches any future stratifier bug whatever its mechanism.

**Post-failure frames.** `post_failure_mask` marks every frame from the moment the robot became unrecoverable. The predicate mirrors the simulators' own crash test — base below 0.15 m, or roll/pitch past 60° — plus a sustained-collapse test (below 0.175 m for 25 consecutive control steps = 0.5 s), because a folded robot rests just above the floor, level, and one pilot logged 26 s of exactly that as a clean episode. Once the predicate first holds, every later frame is flagged. v3 kept 1,625 frames of a tumbling robot with no row-level marker at all.

Failure episodes are **kept** (`store_failed_episodes = True`): the steps leading into a fall are where contact and GRF estimates break down. `post_failure` and `valid` travel on every index row so they can be filtered or stratified without joining back to the episode table. A manual respawn is never stored.

**Episode boundaries.** `episode_mode = "event"` closes an episode on a task event, with `episode_duration_s = 60.0` as a safety cap. Under navigation, `end_episode_on_goal = True` closes at a goal — a real `terminate_by="success"` — but only after `goal_closes_episode_after_s = 30.0`, because closing at *every* goal left a 9 s median episode in v3. `min_episode_duration_s = 1.0` is the only length floor and must be at least the longest window any downstream dataset will cut.


## 9. On-disk layout, and the gate

```
<output_root_dir>/<run_folder>/
    episodes/<episode_id>.parquet   per-timestep rows (zstd)
    episodes.parquet                per-episode metadata
    index.parquet                   the complete label index
    index_balanced.parquet          (episode_id, t, split, weight)
    run_metadata.json               rates, noise, contact labeling, coverage, GRF stats
    validation_report.json          every check, written by the gate below
<output_root_dir>/dataset_summary.json     cross-run persistent statistics
<output_root_dir>/_quarantine/<run>/       runs that failed the gate
```

Run folders follow `{prefix}_{robot}_{scene}_{gait}_{timestamp}`. With `save_after_each_episode = True` the episode table and the index are rewritten after every closed episode, so an interrupted run still leaves a usable dataset.

**The run is validated before it counts.** When the run closes, `tools/validate_run.py` runs over it and a failing folder is moved to `_quarantine/`, so nothing downstream picks it up by globbing `datasets/*/`. This exists because the audited run declared `clipping_audit.limit = 0.001`, reported three channels over it, and shipped anyway: the arithmetic was right and the check was right, and nothing ran it. `tools/test_validator.sh` corrupts a folder one field at a time and requires the owning check to fail each time — a check that never fails is not a check.

Key columns on an `index.parquet` row:

| Column | Use |
| --- | --- |
| `episode_id`, `t`, `episode_n_steps` | Where to slice the window from, and whether any `W` fits |
| `bucket_key` | The full bucket label — stratify and weight on this |
| `contact_state`, `contact_bits` | Named pattern and `"1001"`-style bits |
| `perturbation_active`, `perturbation_level`, `external_force_n` | Perturbation axis |
| `speed_bin` | Command regime |
| `grf_total_n`, `grf_load_share_max`, `grf_tangential_ratio_max` | What §7 measures |
| `post_failure`, `valid` | Filter tumbling frames (`failed`, and not-`failed`-or-`severe`) |
| `operating_regime`, `base_height_terrain`, `base_tilt_deg`, `non_foot_contact_n` | §4b — also in the episode files, so a consumer reading those directly need not join |
| `schedule_mismatch`, `schedule_mismatch_sustained`, `schedule_mismatch_bits` | §4 |
| `randomization_group_id`, `seed`, `run_id` | Split join key and provenance |
| `terrain`, `gait`, `terminate_reason` | Episode context |

`datasets/signal_bounds.json` is a **rendering of `go2_constrains.yaml` + `go2_stance.yaml`**, not a separate source: `tools/build_signal_bounds.py` translates them and records both SHA-256s. Bounds are per leg (front feet reach forward, hind feet reach back, and a union of the two wastes most of the range), left/right pairs are asserted to mirror exactly in y so the sagittal-mirror augmentation stays valid, and `sensor_full_scale` lives outside `signals` so it can never reach `ImageEncoder(constraints=...)`. If a bound is wrong, fix the YAML.

The reconciliation report separates the two ways a rail can be wrong, because they need opposite fixes and only one is urgent: **CLIPPING** means the rail is too tight and real samples saturate (information destroyed), **LOW CONTRAST** means it is too wide and the PI quantises the channel away (range wasted). Channels that are knowingly wide sit in `low_contrast_accepted` with a stated reason, so a settled decision stops being re-reported as an open finding. Three are accepted today — `joint_pos.*.HAA`, `joint_pos.*.HFE` and `base_lin_vel.x` — all because narrowing them to fit flat trot would clip on crawl, stairs or reverse, and a too-tight rail is the expensive error.

`dataset_summary.json` is **read and written**, not only written. Path: `<output_root_dir>/dataset_summary.json` (default `datasets/dataset_summary.json`).

| When | What |
| --- | --- |
| Session start (`DatasetBucketSystem.__init__`) | `load_dataset_summary()`. Missing file → empty memory, or bootstrap by scanning existing `index.parquet` folders. Wrong `schema_version` or `compat_key` → old file rotated aside (`dataset_summary.v<N>.json` or `dataset_summary_<oldkey>.json`), never deleted, and a fresh aggregate starts. |
| After each saved episode (`save_after_each_episode`) and again at shutdown | `update_dataset_summary` **re-reads** the JSON, merges this run (keyed on the absolute `index.parquet` path, so a second write of the same run is idempotent), writes atomically. |

It carries a **`compat_key`**: a hash of the episode schema version, the signal-bounds version, the contact-labelling thresholds, the bucket key fields and the GRF definition. Runs may only be summed when it matches — a changed bucket key renames every bucket and a changed threshold changes what `contact_state` *means*, so an aggregate spanning either describes nothing.

### What the console prints when the sim ends

`finish()` (atexit / viewer close) prints specs for **this collection job**, then writes the run and updates the JSON. The dump is `print_summary()` plus the recorder's quality lines — it is **not** a reprint of the all-runs `summary` block in `dataset_summary.json`.

`DatasetBucketSystem.print_summary()`:

- samples stored / seen, perturbation count and ratio (vs `min_perturbation_ratio`), rare-contact count, active buckets, episode count
- episodes per `terminate_by`, samples per split
- contact-state histogram
- worst five bucket-group perturbation ratios
- pointer to `grf_diversity` (no GRF number here — §7)
- `undercollected_buckets` / `underpopulated_buckets` warnings

Then the recorder, still for this run:

- contact quality: 1-step-run fraction (limit 2%) and transitions/foot/s (limit 3.5)
- clipping audit (worst channel vs `CLIPPING_LIMIT`)
- operating-regime mix; non-crash episodes that spent >10% of frames degraded
- coverage warnings; stance GRF median / p99 / max
- path of the written run, then `tools/validate_run.py` (§9 gate)

Combined totals across previous folders live in `dataset_summary.json` (`summary` + per-run `datasets`). They are updated on disk at the same shutdown; they are not printed as a second “whole corpus” report. For that JSON:

```
python -m mpx.utils.dataset_collection.dataset_summary_plotter datasets/dataset_summary.json --show
```


## 10. Gait labels must come from the controller

`gait_from_phase_offsets(timer_t)` reads the controller's per-leg phase offsets and matches them against the reference patterns:

| Gait | FL FR RL RR offsets |
| --- | --- |
| `trot` | 0.5, 0.0, 0.0, 0.5 (diagonal pairs) |
| `pace` | 0.5, 0.0, 0.5, 0.0 (lateral pairs) |
| `bound` | 0.5, 0.5, 0.0, 0.0 (front/hind pairs) |
| `crawl` | 0.25, 0.75, 0.0, 0.5 (one leg at a time) |

Matching is done over every whole-cycle anchor shift (the same gait can be written with any leg as phase zero) with circular distance (0.98 and 0.00 are 0.02 apart) and tolerance 0.05. An unrecognised pattern falls back to `TRANSITION` rather than guessing.

The comparison is **element-wise, never sorted**: sorting discards *which* leg holds which phase, and trot `[0.5, 0, 0, 0.5]` and pace `[0.5, 0, 0.5, 0]` sort to the same multiset.

This function exists because the gait label was once hardcoded. The audited run set `GaitType.TROT` in the simulator while `config_go2.timer_t` held a crawl pattern, so every episode — and every run folder name — claimed a gait the robot never walked.


## 11. Quick reference — tuning knobs

All in `mpx/config/sim_config/config_dataset_bucket.py`. Nothing below is also hardcoded anywhere else; two of these used to be defined both here and as literals inside `print_summary`, and they agreed until someone tuned one.

**`DatasetBucketConfig`**

| Knob | Default | Effect |
| --- | --- | --- |
| `bucket_capacity` | 5,000 | Reference population only; buckets never evict |
| `perturbation_small_bw_frac` | 0.10 | `none` → `small` boundary, as a fraction of body weight |
| `perturbation_large_bw_frac` | 0.35 | `small` → `large` boundary |
| `min_perturbation_ratio` | 0.25 | Target perturbed fraction |
| `min_bucket_samples_for_ratio_check` | 100 | Floor for reporting a per-bucket ratio |
| `class_balance_alpha` | 0.5 | Weight exponent |
| `min_bucket_samples_for_weighting` | 50 | Below it: weight 0, reported as not collected |
| `max_weight_ratio` | 10.0 | Weight clip, then renormalise to mean 1.0 |
| `weighted_splits` | `("train",)` | Splits the weights apply to |
| `load_share_iqr_warn` | 0.02 | Low load-distribution diversity |
| `total_grf_iqr_warn_bw_frac` | 0.08 | Low total-GRF spread, body-weight relative |
| `min_randomization_groups_per_bucket` | 3 | Single-condition bucket |
| `impact_dominated_p99_ratio` | 2.5 | Spread is only touchdown impacts |
| `diversity_min_bucket_samples` | 100 | Floor for any diversity statistic |
| `underpopulated_share_of_expected` | 0.25 | Relative population warning |
| `mismatch_edge_radius` | 2 | Steps from a transition that count as jitter |
| `mismatch_sustained_min_run` | 3 | Frames that make a mismatch run sustained |
| `perturbation_force_range_n` | (10, 50) N | Sampler range the `small`/`large` boundary splits |

**`OperatingRegimeConfig`** (rails also readable from the `orientation` block of `go2_constrains.yaml`)

| Knob | Default | Effect |
| --- | --- | --- |
| `height_degraded_m` / `_severe_m` / `_failed_m` | 0.195 / 0.175 / 0.150 | Terrain-relative base height |
| `tilt_degraded_deg` / `_severe_deg` / `_failed_deg` | 12 / 25 / 45 | Angle from gravity |
| `non_foot_contact_force_n` | 5.0 | Above it, at least `severe`, dwell filter bypassed |
| `tracking_error_degraded_mps` | 0.35 | ‖realised − commanded‖ planar velocity |
| `dwell_steps` | 3 | Posture signals must hold this long to change the regime |

**Contact labelling** (`ContactLabelConfig`, in `contact_labeling.py` — the only place a contact threshold exists)

| Knob | Default | Effect |
| --- | --- | --- |
| `on_threshold_n` | 15.0 N | Touchdown rail |
| `off_threshold_n` | 5.0 N | Release rail |
| `min_dwell_steps` | 3 | 60 ms at 50 Hz |
| `substep_majority` | 0.5 | Fraction of substeps that must be loaded |

**Rates, episodes, export, output**

| Knob | Default | Effect |
| --- | --- | --- |
| `sim_hz` / `control_hz` | 500 / 50 | 10 substeps per labelled step |
| `label_stride` | 1 | Index every timestep |
| `episode_mode` | `"event"` | Close on task event; duration is a cap |
| `goal_closes_episode_after_s` | 30.0 | Earliest goal that ends an episode |
| `randomize_per_episode` | True | Resample DR at every episode boundary |
| `segmented_commands` | True | Velocity-command driving instead of goal following |
| `val_ratio` / `test_ratio` / `split_seed` | 0.15 / 0.10 / 0 | Per-episode split |
| `validate_after_run` | True | Run the validator when the run closes |
| `validate_skip` | `()` | Nothing skipped — the difficulty probe is the gate that decides whether a folder is worth training on, and it spent every run so far reported as SKIPPED |
| `quarantine_failed_runs` | True | Move a failing run to `_quarantine/` |
