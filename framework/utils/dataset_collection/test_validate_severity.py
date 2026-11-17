"""Integrity vs coverage: only integrity failures quarantine a run."""

from __future__ import annotations

import inspect

from tools.validate_run import CHECKS, check_severity, run_checks

COVERAGE = {
    "command_coverage",
    "speed_range_covered",
    "episodes_long_enough",
    "no_channel_clipping",
    "operating_regime_separates_crashes",
    "attitude_estimator_error_recorded",
    "height_label_agrees_with_grf_label",
    "task_not_saturated",
}


def test_coverage_checks_are_tagged():
    by_name = {n: s for n, _fn, s in CHECKS}
    assert set(COVERAGE) <= set(by_name)
    for name, sev in by_name.items():
        assert sev == ("coverage" if name in COVERAGE else "integrity"), name


def test_run_checks_counts_only_integrity(tmp_path):
    report, n_failed = run_checks(tmp_path, write_report=False, verbose=False)
    n_integrity = sum(1 for _n, _fn, s in CHECKS if s == "integrity")
    assert n_failed == n_integrity
    for name, _fn, sev in CHECKS:
        assert report[name].startswith("FAIL")
        if sev == "coverage":
            assert check_severity(name) == "coverage"


def test_contact_chatter_does_not_gate_gait_rate():
    fn = next(f for n, f, _s in CHECKS if n == "contact_labels_not_chattering")
    src = inspect.getsource(fn)
    assert "fraction_one" in src
    assert "transitions/foot/s" not in src
