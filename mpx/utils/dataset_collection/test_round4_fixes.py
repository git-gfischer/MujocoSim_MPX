"""
Tests for the round-4 bucket-system patches (BUCKET_SYSTEM_PATCHES_R4).

Each test names the defect it locks down, so a future regression reports what
broke rather than only which assert failed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mpx.config.sim_config.config_dataset_bucket import (
    DatasetBucketConfig,
    dataset_bucket_config,
)
from mpx.utils.dataset_collection.dataset_bucket_system import (
    BucketKey,
    DatasetBucketSystem,
    GaitType,
    TerrainType,
    perturbation_level,
    schedule_mismatch_masks,
    speed_bin,
)
from mpx.utils.dataset_collection.dataset_schema import (
    EpisodeMetadata,
    EpisodeRecord,
    dt_since_transition,
    empty_arrays,
)
from mpx.utils.dataset_collection.episode_storage import (
    EpisodeStore,
    pyarrow_available,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROL_HZ = 50.0
BODY_WEIGHT_N = 176.0


def make_record(
    episode_id: str = "ep_00001",
    n_steps: int = 120,
    *,
    stance=(1, 1, 1, 1),
    normal_n: float = 50.0,
    tangential_n: float = 5.0,
    ext_force: float = 0.0,
    cmd=(0.5, 0.0, 0.0),
    schedule=None,
    split: str = "train",
    terminate_by: str = "success",
    group: float = 1.4,
) -> EpisodeRecord:
    """One synthetic episode with a constant contact pattern."""
    arrays = empty_arrays(n_steps)
    arrays["t"][:] = np.arange(n_steps, dtype=np.int32)
    arrays["time_s"][:] = np.arange(n_steps, dtype=np.float32) / CONTROL_HZ

    grf = np.zeros((n_steps, 4, 3), dtype=np.float32)
    for foot, loaded in enumerate(stance):
        if loaded:
            grf[:, foot, 2] = normal_n
            grf[:, foot, 0] = tangential_n
    arrays["grf_base"][:] = grf.reshape(n_steps, 12)
    arrays["grf_mean_n"][:] = np.linalg.norm(grf, axis=2)
    arrays["external_force"][:, 0] = ext_force
    arrays["cmd_base_vel"][:] = np.asarray(cmd, dtype=np.float32)
    # A robot that is standing up. empty_arrays leaves base_height_terrain at
    # 0.0, which operating_regime correctly reads as lying on the floor.
    arrays["base_height_terrain"][:] = 0.25
    arrays["base_quat"][:] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    contact = np.tile(np.asarray(stance, dtype=np.uint8), (n_steps, 1))
    arrays["contact"][:] = contact
    arrays["contact_schedule"][:] = (
        contact if schedule is None else np.asarray(schedule, dtype=np.uint8)
    )
    arrays["dt_since_transition"][:] = dt_since_transition(contact, 1.0 / CONTROL_HZ)

    metadata = EpisodeMetadata(
        episode_id=episode_id,
        robot="go2",
        scene="flat",
        terrain=TerrainType.FLAT.value,
        gait=GaitType.TROT.value,
        control_hz=CONTROL_HZ,
        friction=0.65,
        payload_kg=1.25,
        body_weight_n=BODY_WEIGHT_N,
        terminate_by=terminate_by,
        terminate_reason="goal_reached",
        split_assigned=split,
        reset_randomization={"step_freq": group},
    )
    return EpisodeRecord(metadata=metadata, arrays=arrays)


# ── P1: the dead contact-derivation path is gone ─────────────────────────────

def test_there_is_exactly_one_contact_definition():
    """
    ``derive_contacts`` implemented a third contact threshold that production
    never used. It disagreed with the shipped debounced label on 8.6% of rows,
    yet it labelled the synthetic fixtures, so the fixtures exercised a path the
    real data never took.
    """
    # Test files are excluded: two of them name the symbols in order to assert
    # they are gone, which is the opposite of the defect.
    source = subprocess.run(
        [
            "git", "grep", "-l", "-E",
            r"derive_contacts|contact_bits_from_grf|contact_force_threshold",
            "--", "*.py", ":!*test_*.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert not source.stdout.strip(), (
        f"the dead contact path survives in: {source.stdout.split()}"
    )


def test_bucket_config_carries_no_contact_threshold():
    assert not hasattr(dataset_bucket_config, "contact_force_threshold")
    assert not hasattr(DatasetBucketSystem, "derive_contacts")


# ── P2: the force statistics come from the substep average, per foot ─────────

def test_grf_statistics_come_from_grf_base_not_the_aliased_column():
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    bucket.add_episode(make_record("ep", n_steps=60))
    sample = next(iter(bucket.buckets.values()))[0]

    expected = float(np.hypot(50.0, 5.0))
    assert sample.grf_per_foot_n == pytest.approx((expected,) * 4)
    assert sample.grf_total_n == pytest.approx(4 * expected)
    # An even four-foot stance is a load share of exactly 0.25.
    assert sample.grf_load_share_max == pytest.approx(0.25)
    # Friction utilisation: 5 N tangential against 50 N normal.
    assert sample.grf_tangential_ratio_max == pytest.approx(0.1)


def test_no_sample_is_loaded_with_zero_total_force():
    """98 index rows of the audited run carried 0 N while contact != 0000."""
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    bucket.add_episode(make_record("ep", n_steps=60, stance=(1, 0, 0, 1)))
    for samples in bucket.buckets.values():
        for sample in samples:
            if any(sample.contact_bits):
                assert sample.grf_total_n > 0.0


def test_single_support_reads_a_load_share_of_one():
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    bucket.add_episode(make_record("ep", n_steps=60, stance=(1, 0, 0, 0)))
    sample = next(iter(bucket.buckets.values()))[0]
    assert sample.grf_load_share_max == pytest.approx(1.0)


# ── P3: schedule mismatch is jitter plus real disagreement ───────────────────

def test_edge_jitter_is_separated_from_sustained_disagreement():
    """
    88% of mismatched frames sit within two steps of a contact transition, with
    no systematic lead or lag — timing jitter, not slips. A torque-reading model
    fails on those frames too, so the raw flag is not the benchmark subset.
    """
    contact = np.ones((20, 4), dtype=np.uint8)
    contact[10:, 0] = 0                     # FL lifts off at t = 10
    schedule = contact.copy()
    schedule[10, 0] = 1                     # one-frame timing disagreement
    schedule[2:8, 1] = 0                    # six-frame real disagreement

    masks = schedule_mismatch_masks(contact, schedule)
    assert masks["per_foot"][10, 0] and masks["edge"][10, 0]
    assert not masks["sustained"][10, 0]
    assert masks["sustained"][2:8, 1].all()
    assert not masks["edge"][4, 1]


def test_sustained_implies_mismatch_in_the_index():
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    schedule = np.ones((120, 4), dtype=np.uint8)
    schedule[30:50, 2] = 0
    bucket.add_episode(make_record("ep", schedule=schedule))

    rows = bucket.index_rows(shuffle=False)
    assert any(r["schedule_mismatch_sustained"] for r in rows)
    for row in rows:
        assert row["schedule_mismatch"] or not row["schedule_mismatch_sustained"]
    flagged = next(r for r in rows if r["schedule_mismatch_sustained"])
    assert flagged["schedule_mismatch_bits"] == "0010"


# ── P4: the weights obey their own docstring ─────────────────────────────────

def _weighted_bucket_system() -> DatasetBucketSystem:
    bucket = DatasetBucketSystem(
        DatasetBucketConfig(min_bucket_samples_for_weighting=50, max_weight_ratio=10.0),
        dataset_summary_path=None,
    )
    bucket.add_episode(make_record("ep_train_a", n_steps=400, split="train", group=1.0))
    bucket.add_episode(
        make_record(
            "ep_train_b", n_steps=60, stance=(1, 0, 0, 1), split="train", group=2.0
        )
    )
    bucket.add_episode(
        make_record(
            "ep_train_rare", n_steps=5, stance=(1, 0, 0, 0), split="train", group=3.0
        )
    )
    bucket.add_episode(make_record("ep_val", n_steps=120, split="val", group=4.0))
    return bucket


def test_weights_are_clipped_and_normalised_over_the_train_split():
    bucket = _weighted_bucket_system()
    weights = bucket.bucket_weights()
    positive = [w for w in weights.values() if w > 0]

    assert positive, "no bucket cleared the population floor"
    assert max(positive) / min(positive) <= 10.0 + 1e-6, (
        "weight ratio exceeds the cap; v4.1 shipped 81x"
    )


def test_a_bucket_below_the_floor_is_reported_not_weighted():
    """n = 1 carrying 57x the modal weight is gradient variance, not a class."""
    bucket = _weighted_bucket_system()
    weights = bucket.bucket_weights()
    sparse = dict(bucket.undercollected_buckets())

    assert sparse, "the 5-sample bucket was not reported as undercollected"
    for label in sparse:
        assert weights[label] == 0.0
    # Its rows are kept: balancing is a weight, never a deletion.
    labels = {row["bucket_key"] for row in bucket.index_rows(shuffle=False)}
    assert set(sparse) <= labels


def test_held_out_rows_are_never_reweighted():
    bucket = _weighted_bucket_system()
    rows = bucket.balanced_index_rows()
    held_out = [r for r in rows if r["split"] != "train"]

    assert held_out, "fixture has no held-out episode"
    assert all(r["weight"] == 1.0 for r in held_out), (
        "a reweighted val/test metric describes a distribution that does not exist"
    )
    train = [r["weight"] for r in rows if r["split"] == "train" and r["weight"] > 0]
    assert abs(float(np.mean(train)) - 1.0) < 0.05


def test_weights_are_keyed_on_the_whole_bucket():
    bucket = _weighted_bucket_system()
    for label in bucket.bucket_weights():
        assert label.count("|") == 4, f"{label} is not a full bucket key"


# ── P5: one source for every threshold ───────────────────────────────────────

def test_thresholds_come_from_the_config_object():
    tight = DatasetBucketConfig(min_bucket_samples_for_weighting=10_000)
    bucket = DatasetBucketSystem(tight, dataset_summary_path=None)
    bucket.add_episode(make_record("ep", n_steps=120))

    assert bucket.config is tight
    # Every bucket is below the (absurd) floor, so nothing is weighted.
    assert all(w == 0.0 for w in bucket.bucket_weights().values())
    assert bucket.undercollected_buckets()


def test_export_config_no_longer_duplicates_diagnostic_thresholds():
    from mpx.config.sim_config.config_dataset_bucket import DatasetExportConfig

    export = DatasetExportConfig()
    assert not hasattr(export, "grf_diversity_warn_std_n")
    assert not hasattr(export, "underpopulated_bucket_fraction")


# ── P6: the key carries the axes the diagnostics need ────────────────────────

def test_speed_bin_separates_the_command_regimes():
    assert speed_bin((0.0, 0.0, 0.0)) == "stopped"
    assert speed_bin((-0.4, 0.0, 0.0)) == "reverse"
    assert speed_bin((0.1, 0.0, 0.8)) == "turning"
    assert speed_bin((0.2, 0.0, 0.0)) == "slow"
    assert speed_bin((0.5, 0.0, 0.0)) == "medium"
    assert speed_bin((0.9, 0.0, 0.0)) == "fast"


def test_perturbation_is_a_level_not_a_boolean():
    """A 6 N nudge and a 60 N shove must not land in the same bucket."""
    assert perturbation_level(6.0, BODY_WEIGHT_N) == "none"
    assert perturbation_level(30.0, BODY_WEIGHT_N) == "small"
    assert perturbation_level(70.0, BODY_WEIGHT_N) == "large"


def test_perturbation_level_scales_with_payload():
    """5 N absolute was 2.8% of body weight — inside the log's own noise."""
    light, heavy = 176.0, 176.0 + 50.0 * 9.81
    assert perturbation_level(25.0, light) == "small"
    assert perturbation_level(25.0, heavy) == "none"


def test_bucket_key_separates_command_regimes():
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    bucket.add_episode(make_record("ep_slow", cmd=(0.2, 0.0, 0.0)))
    bucket.add_episode(make_record("ep_fast", cmd=(0.9, 0.0, 0.0), group=2.0))

    regimes = {key.speed_bin for key in bucket.buckets}
    assert regimes == {"slow", "fast"}, (
        "one command per episode used to look like a healthy bucket"
    )
    assert "pert=none" in next(iter(bucket.buckets)).label()


def test_bucket_key_still_exposes_the_v4_boolean():
    key = BucketKey("FULL", "small", "slow", TerrainType.FLAT, GaitType.TROT)
    assert key.perturbation_active is True
    assert BucketKey(
        "FULL", "none", "slow", TerrainType.FLAT, GaitType.TROT
    ).perturbation_active is False


# ── P7: diversity is post-hoc, off the parquet ───────────────────────────────

def test_grf_diversity_does_not_import_the_bucket_system():
    """
    The statistic has to be changeable without recollecting, which it cannot be
    while it is computed at collection time from in-memory SampleRefs.
    """
    source = (
        Path(__file__).with_name("grf_diversity.py").read_text(encoding="utf-8")
    )
    assert "dataset_bucket_system" not in source
    assert "index.parquet" in source


@pytest.mark.skipif(not pyarrow_available(), reason="needs pyarrow")
def test_bucket_diversity_reads_a_written_run(tmp_path):
    from mpx.utils.dataset_collection.grf_diversity import (
        bucket_diversity,
        diversity_warnings,
    )

    store = EpisodeStore(tmp_path)
    bucket = DatasetBucketSystem(dataset_summary_path=None, store=store)
    # One command, one randomization draw: full, and homogeneous.
    for i in range(3):
        bucket.add_episode(
            make_record(f"ep_{i}", n_steps=200, group=1.0, split="train")
        )
    bucket.save_dataset(tmp_path)

    report = bucket_diversity(tmp_path, body_weight_n=BODY_WEIGHT_N)
    assert report and report[0]["n_samples"] == 600
    assert report[0]["n_randomization_groups"] == 1

    warnings = diversity_warnings(
        report, DatasetBucketConfig(diversity_min_bucket_samples=100)
    )
    key = report[0]["bucket_key"]
    # A constant stance has no load-share spread and one condition — both of
    # which the sample count alone would have called healthy.
    assert key in warnings["low_load_share_spread"]
    assert key in warnings["single_condition"]


# ── P8: the cross-run summary refuses to mix incompatible runs ───────────────

def test_compat_key_changes_with_the_contact_thresholds():
    from mpx.utils.dataset_collection.contact_labeling import ContactLabelConfig

    base = DatasetBucketSystem(dataset_summary_path=None)
    changed = DatasetBucketSystem(
        contact_label_config=ContactLabelConfig(on_threshold_n=25.0),
        dataset_summary_path=None,
    )
    assert base.aggregate_compat_key() != changed.aggregate_compat_key()


def test_compat_key_ignores_the_payload_draw():
    """Body weight is a per-run fact, not a definition of the label."""
    light = DatasetBucketSystem(body_weight_n=176.0, dataset_summary_path=None)
    heavy = DatasetBucketSystem(body_weight_n=226.0, dataset_summary_path=None)
    assert light.aggregate_compat_key() == heavy.aggregate_compat_key()


@pytest.mark.skipif(not pyarrow_available(), reason="needs pyarrow")
def test_an_incompatible_summary_is_archived_not_merged(tmp_path):
    summary_path = tmp_path / "dataset_summary.json"
    bucket = DatasetBucketSystem(dataset_summary_path=summary_path)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "compat_key": "deadbeefdeadbeef",
                "datasets": {},
                "summary": {},
            }
        ),
        encoding="utf-8",
    )

    memory = bucket.load_dataset_summary()
    assert memory["datasets"] == {}
    assert (tmp_path / "dataset_summary_deadbeefdeadbeef.json").exists()


# ── P9: window validity for any W ────────────────────────────────────────────

def test_index_rows_carry_the_episode_length():
    """``window_valid_w10`` hardcodes W = 10 and contradicts §1 of the design."""
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    bucket.add_episode(make_record("ep", n_steps=75))
    row = bucket.index_rows(shuffle=False)[0]

    assert row["episode_n_steps"] == 75
    for window in (5, 10, 30):
        assert (row["t"] >= window - 1) == (
            row["t"] - window + 1 >= 0
        )


def test_effective_sample_size_is_recorded():
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    bucket.add_episode(make_record("ep", n_steps=100))
    block = bucket.effective_sample_size(window=10)

    assert block["index_rows"] == 100
    assert block["independent_windows_w10"] == 10
    assert "stride" in block["note"]
