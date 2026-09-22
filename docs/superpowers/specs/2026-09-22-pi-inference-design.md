# PI GNN inference (sim-ready class + dataset bench)

Date: 2026-09-22

## Goal

Replace the unused concatCNN `_legacy/inference.py` with a `pi_gnn` / `pi_vgnn` inference module that (1) can be imported later by `mpx/addons/ProprioceptiveImage/mpx_sim/quad_locomotion.py` and (2) can run alone against a checkpoint and a parquet dataset using the same metrics as training.

## Why

Training already has `get_model`, `dataset_train_val`, and `evaluate()`. There is no load-weights-and-run path for the live heads. The quarantined inference file talks JPEG 16-class images and must not be wired back.

Default GRF ablation is Z-only (`force_dim: 1`). XYZ is a later ablation (`force_dim: 3`). Inference does not choose a GRF mode; it reports whatever the loaded head emits.

## Decision

One file `PI_NN/inference.py`: class `PIInference` plus a CLI. Training yaml gains `model.force_dim`. `get_model` passes it into `MultiTaskFootHead`. `collate_pi` slices the Z column when `force_dim==1`.

Rejected: CLI-only script (sim would re-copy load/forward). Rejected: an Inference package (Tester, GradCAM, filters). Rejected: always slicing Z off an XYZ head (a `force_dim: 3` checkpoint must stay `(B, 4, 3)`).

Sim loop wiring is not this spec. `batch_from_pi` is the hook.

## Yaml

Under `model` in `PI_NN/config/NN_config.yaml`:

```yaml
model:
  predict_force: True
  force_dim: 1          # 1 = Z only (default); 3 = XYZ
  predict_ext_force: False
```

`get_model` reads `int(mcfg.get("force_dim", 1))`. Only `1` or `3`; anything else is `ValueError`. Passed only into `MultiTaskFootHead` (spatial head has no GRF branch). Missing key defaults to `1`, which will not load an existing XYZ checkpoint.

`predict_ext_force` stays as today: builds `ExternalForceHead` when True. No ext-force loss in this spec. No base-velocity head.

## API

```python
inf = PIInference(cfg_path, weights_path)   # eval, yaml must match checkpoint

out = inf.predict(batch)   # tensors moved to the model device
# batch keys: joint, torque, foot_pos, foot_vel, imu  (same as collate_pi)
# out = model dict plus:
#   contact_prob (B, 4)  sigmoid of contact_logits
#   contact      (B, 4)  {0, 1} at threshold 0.5
# grf is (B, 4, force_dim) when the force branch exists
# ext_force is (B, 3) only if that head is in the checkpoint

out = inf.predict(inf.batch_from_pi(pi))    # later sim, after get_PI_image()
```

`batch_from_pi(pi)` packs the stacked arrays already on `ProprioceptiveImage` after a successful `get_PI_image()`:

| source | batch key | notes |
|---|---|---|
| `_joint_images` `(12, 6, W, W)` | `joint` | skip if `None` |
| `_torque_images` `(12, 4, W, W)` | `torque` | skip if `None` |
| `_footpos_images` `(4, 4, W, W)` | `foot_pos` | skip if `None` |
| `_footvel_images` `(4, 4, W, W)` | `foot_vel` | skip if `None` |
| `_trunk_image` `(L, L, 5)` | `imu` | permute to `(1, 5, L, L)` |

Each present array gets a leading batch dim of 1. Foot order is graph order LF, RF, LH, RH (same as the dataset).

CLI (cwd / `sys.path` like `Train.py`):

```bash
python inference.py --cfg config/NN_config.yaml --weights path/to/best.pth --split test
```

Loads the split via `dataset_train_val`, builds `get_loss_fn(cfg)` (same as `Train.py`), runs `Metrics.wandb_logging.evaluate` with that criterion, prints the metric dict, exits. Default `--split test`; if that loader is `None`, use `val`. No latency file, no prediction dump.

## Dataset / loss / metrics

`collate_pi(samples, force_dim=1)` still reshapes `grf_base` `(12,)` → `grf` `(B, 4, 3)`. When `force_dim==1`, keep `grf[..., 2:3]` so the target is `(B, 4, 1)`. `force_dim==3` leaves `(B, 4, 3)`. `dataset_train_val` passes yaml `force_dim` via `functools.partial`. The existing collate self-check that asserts `(B, 4, 3)` must pass `force_dim=3`.

Loss is unchanged: Huber on `force_conditional` vs `targets["grf"]` with matching last dim.

Metrics already branch on `D = gp.shape[-1]`. Today `D==1` is labelled axis `n`. Change that label to `z`. XYZ still uses `x/y/z`.

## Checkpoints

Yaml must match the weights. A Z-only `best.pth` will not `load_state_dict` with `force_dim: 3`, and the reverse. No converter. XYZ ablation = new run with `force_dim: 3`. Inference has no GRF mode flag; `out["grf"].shape[-1]` is the mode.

## Errors

- `force_dim` not in `{1, 3}` → `ValueError` from `get_model`
- missing weights file → `FileNotFoundError`
- yaml `force_dim` ≠ checkpoint head → `load_state_dict` fails
- missing image key in `predict` → the model's existing error
- `batch_from_pi` before a successful `get_PI_image()` (all stacked arrays `None`) → `ValueError`

## Check

- `collate_pi(..., force_dim=1)` → `grf` is `(B, 4, 1)` equal to the Z column of the `(B, 4, 3)` reshape; `force_dim=3` stays `(B, 4, 3)`
- `get_model` with `force_dim: 1` → `network.head.force_dim == 1`
- dummy `predict`: `contact` in `{0,1}`, `contact_prob` in `[0,1]`, `grf.shape[-1] == force_dim`
- `pi_metrics` `D=1` axis name is `z` (update the existing self-check if it assumes three axes)

## Out of scope

- Wiring `mpx_sim/quad_locomotion.py` or `mpx/simulators/quadruped/quad_locomotion.py`
- Base-velocity head / loss
- External-force loss (head still optional; inference returns `ext_force` if present)
- Latency, prediction dumps, GradCAM, post-process filters
- Legacy concatCNN / 16-class path
