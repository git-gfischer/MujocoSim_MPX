# Per-modality proprioceptive image encoders

Date: 2026-09-15

## Goal

Replace the shared `ConvStem` / `NodeSpatialEncoder` in
`mpx/addons/ProprioceptiveImage/PI_NN/modules/pi_enc.py` with five named
encoders (torque, IMU, joints, foot position, foot velocity). Each encoder maps
one family's tiles to independent feature vectors. There is no adaptive pooling.
The GNN concatenates those vectors onto nodes in a later change; this spec does
not rewire `pi_gnn.py` beyond unblocking imports.

## Why not one stem

The five images do not share channel count or spatial size:

| Family | Channels | Spatial |
|---|---|---|
| Torque | 4 | `w × w` |
| Joint | 1 | `w × w` |
| Foot pos | 3 | `w × w` |
| Foot vel | 2 | `w × w` |
| IMU | 3 | `64 × 64` (fixed) |

`w` is configured once (default **30**). Adaptive pooling was previously used to
force 60×60 and 64×64 onto one grid; that resampling is not used here. Fusion
across joints/feet/modalities is the GNN's job, not the encoder's.

## Architecture

One private backbone, five public encoder classes, one optional holder that
takes `image_size` once.

```
PIEncoders(image_size=30, torque_dim, joint_dim, foot_pos_dim, foot_vel_dim, imu_dim)
  ├── TorqueEncoder     StridedMapEncoder (2 strides) + concat 4 side scalars
  ├── JointEncoder      StridedMapEncoder (2 strides)
  ├── FootPosEncoder    StridedMapEncoder (2 strides)
  ├── FootVelEncoder    StridedMapEncoder (2 strides)
  └── IMUEncoder        StridedMapEncoder (3 strides, in_size=64)
```

`StridedMapEncoder` is an implementation detail. Callers use the named classes
or `PIEncoders`.

Layout is PyTorch NCHW. A family with several tiles is folded with existing
`_apply_nodewise`: `(B, N, C, H, W)` → conv on `(B·N, C, H, W)` → `(B, N, dim)`.
Weights are shared across `N`. Tiles do not mix: encoding 12 tiles in one call
equals 12 single-tile calls with the same weights.

## Backbone (`StridedMapEncoder`)

Constructor:

- `in_channels`, `in_size`, `n_stride`, `out_dim`
- `mid_channels=16`, `bottleneck_channels=4`

Block, repeated `n_stride` times:

1. `Conv2d(k=3, stride=2, padding=1, bias=False)`
2. GroupNorm (`_gn`)
3. `LeakyReLU(0.01, inplace=True)`

First conv maps `in_channels → mid_channels`; later stride convs stay at
`mid_channels`. Then:

4. `Conv2d(1×1, mid_channels → bottleneck_channels, bias=False)` + GroupNorm + LeakyReLU
5. Flatten
6. `Linear(bottleneck * H' * W', out_dim)`

No `AdaptiveAvgPool`, `AvgPool`, `MaxPool`, or other pooling modules.

Spatial size after one stride-2 pad-1 3×3 conv:

```
H' = floor((H - 1) / 2) + 1
```

Default `w=30`: `30 → 15 → 8`, flatten `4 × 8 × 8 = 256`.
IMU: `64 → 32 → 16 → 8`, flatten `4 × 8 × 8 = 256`.
Alternate `w=17` (tests): `17 → 9 → 5`, flatten `4 × 5 × 5 = 100`.

The linear layer is sized at init from `in_size` and `n_stride`. Changing `w`
does not change `out_dim`.

Forward of the backbone: `(M, C, H, W) → (M, out_dim)`.

## Named encoders

Shared rules:

- `out_dim` is per-modality (separate lengths; not one global feature dim).
- Default `out_dim=32` on each named class so tests can construct without
  inventing GNN widths. `PIEncoders` takes the five dims explicitly (same
  default 32). For torque, `out_dim` / `torque_dim` is the linear width
  **before** side concat; the returned tensor is `out_dim + 4`.
- Rank rule for every named encoder: 5D `(B, N, C, H, W)` → `(B, N, dim)`;
  4D `(B, C, H, W)` → `(B, dim)`.

| Class | `in_channels` | `in_size` | `n_stride` | Extra input | Output dim |
|---|---|---|---|---|---|
| `TorqueEncoder` | 4 | `w` | 2 | `side (B, N, 4)` | `out_dim + 4` |
| `JointEncoder` | 1 | `w` | 2 | — | `out_dim` |
| `FootPosEncoder` | 3 | `w` | 2 | — | `out_dim` |
| `FootVelEncoder` | 2 | `w` | 2 | — | `out_dim` |
| `IMUEncoder` | 3 | 64 | 3 | — | `out_dim` |

Torque: run backbone, then `torch.cat([vec, side], dim=-1)`. Side scalars are
not convolved and not transformed. Last dim of `side` is always 4. Rank of
`side` matches the images: `(B, 4)` with 4D images, `(B, N, 4)` with 5D.

`PIEncoders`:

- `__init__(image_size=30, torque_dim=32, joint_dim=32, foot_pos_dim=32, foot_vel_dim=32, imu_dim=32)`
- `forward(batch)` keys in: `torque`, `torque_side`, `joint`, `foot_pos`, `foot_vel`, `imu`
- keys out: `torque`, `joint`, `foot_pos`, `foot_vel`, `imu` (torque already includes side)

## Error handling

Init:

- `image_size >= 3`
- every `*_dim > 0`
- IMU size is not a constructor argument

Forward `ValueError` when:

- rank is not 4 or 5 as allowed above
- `C`, `H`, or `W` mismatch that encoder
- `torque` / `torque_side` disagree on `B` or `N`
- `torque_side` last dim ≠ 4
- torque is encoded without `side`

`PIEncoders.forward` raises `KeyError` if a required image key is missing.

## What is removed

Delete `ConvStem` and `NodeSpatialEncoder` from `pi_enc.py`.

`pi_gnn.py` currently does `from modules.pi_enc import ConvStem, NodeSpatialEncoder`
and constructs `NodeSpatialEncoder` inside `ProprioceptiveConvGNN`. Remove those
symbols so the file still **imports**. `ProprioceptiveConvGNN.forward` is
expected to fail until a follow-up wires the new vectors onto nodes. Do **not**
implement that concat or scatter here.

## Testing

`mpx/addons/ProprioceptiveImage/PI_NN/modules/test_pi_enc.py`. Real modules, no
mocks. Run with `PI_NN` on `PYTHONPATH` (same as the existing `from utils.conv_utils`
import).

1. Default `w=30`: output shapes match the table (torque is `out_dim + 4`).
2. Batched 12-tile encode equals 12 single-tile encodes with identical weights.
3. Walk modules: no pooling types present.
4. Intermediate maps: `30 → 15 → 8` and IMU `64 → 32 → 16 → 8`.
5. Torque output is `[linear_out | side]` with `side` unchanged.
6. Wrong `C`/`H`/`W`, missing `torque_side`, missing dict key all raise.
7. `w=17` still emits `out_dim` (flatten 100 internally).

## Out of scope

- GNN concatenation and assignment of vectors to nodes
- Changing how images are generated (`torque_img.py`, dataset packing)
- Dropout
- Per-joint (not shared) encoder weights

## Files

- `mpx/addons/ProprioceptiveImage/PI_NN/modules/pi_enc.py` (replace)
- `mpx/addons/ProprioceptiveImage/PI_NN/modules/test_pi_enc.py` (new)
- `mpx/addons/ProprioceptiveImage/PI_NN/modules/pi_gnn.py` (import / stem construction only, so the file still imports)
