"""Unit tests for src/review/ package (M7)."""

import json

import pytest

from src.review.config import MAX_ITERATIONS, PLATEAU_STALE_THRESHOLD, PLATEAU_CONSECUTIVE_MIN
from src.review.schemas import (
    ReviewDecision,
    PrimaryMetric,
    RiskFlag,
    VALID_VERDICTS,
    VALID_ROUTES,
    VALID_STATUSES,
)
from src.review.validator import validate_review_decision, ReviewValidationError
from src.review.history import load_run_history, summarise_history
from src.review.comparator import compute_deltas, compute_trend, find_best_iteration
from src.review.plateau import check_plateau
from src.review.writer import build_record, append_review_decision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(iteration=1, verdict="insufficient", route="continue", **overrides):
    """Helper to build a valid review-decision dict."""
    base = {
        "iteration": iteration,
        "timestamp": "2026-04-16T10:00:00+00:00",
        "status": "completed",
        "plan_summary": f"Iteration {iteration} plan",
        "primary_metric": {"name": "auc_roc", "value": 0.80 + iteration * 0.01, "delta": None if iteration == 1 else 0.01},
        "model_family": "logistic_regression",
        "reviewer_verdict": verdict,
        "reviewer_reasoning": f"Reasoning for iteration {iteration}",
        "router_decision": route,
        "router_reasoning": f"Route reasoning for iteration {iteration}",
        "risk_flags_summary": [],
        "best_iteration": iteration,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        assert MAX_ITERATIONS == 10
        assert PLATEAU_STALE_THRESHOLD == 0.005
        assert PLATEAU_CONSECUTIVE_MIN == 3


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_valid_sets(self):
        assert "sufficient" in VALID_VERDICTS
        assert "insufficient" in VALID_VERDICTS
        assert "continue" in VALID_ROUTES
        assert "rollback" in VALID_ROUTES
        assert "pivot" in VALID_ROUTES

    def test_review_decision_dataclass(self):
        pm = PrimaryMetric(name="auc_roc", value=0.835, delta=None)
        rd = ReviewDecision(
            iteration=1,
            timestamp="2026-04-16T10:00:00Z",
            status="completed",
            plan_summary="Baseline",
            primary_metric=pm,
            model_family="logistic_regression",
            reviewer_verdict="insufficient",
            reviewer_reasoning="Room for improvement",
            router_decision="continue",
            router_reasoning="First iteration",
            risk_flags_summary=[],
            best_iteration=1,
        )
        d = rd.to_dict()
        assert d["iteration"] == 1
        assert d["primary_metric"]["name"] == "auc_roc"
        assert d["reviewer_verdict"] == "insufficient"

    def test_risk_flag_dataclass(self):
        rf = RiskFlag(type="overfitting", severity="medium", evidence="Train/val gap > 10%")
        assert rf.type == "overfitting"


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def test_valid_record(self):
        record = _make_record()
        summary = validate_review_decision(record)
        assert summary["iteration"] == 1
        assert summary["reviewer_verdict"] == "insufficient"

    def test_missing_key(self):
        record = _make_record()
        del record["reviewer_verdict"]
        with pytest.raises(ReviewValidationError, match="Missing required key"):
            validate_review_decision(record)

    def test_invalid_verdict(self):
        record = _make_record(verdict="maybe")
        with pytest.raises(ReviewValidationError, match="reviewer_verdict"):
            validate_review_decision(record)

    def test_invalid_route(self):
        record = _make_record(route="restart")
        with pytest.raises(ReviewValidationError, match="router_decision"):
            validate_review_decision(record)

    def test_invalid_iteration(self):
        record = _make_record(iteration=0)
        with pytest.raises(ReviewValidationError, match="positive int"):
            validate_review_decision(record)

    def test_invalid_status(self):
        record = _make_record(status="running")
        with pytest.raises(ReviewValidationError, match="status"):
            validate_review_decision(record)

    def test_empty_plan_summary(self):
        record = _make_record(plan_summary="")
        with pytest.raises(ReviewValidationError, match="plan_summary"):
            validate_review_decision(record)

    def test_invalid_metric_missing_name(self):
        record = _make_record()
        record["primary_metric"] = {"value": 0.8, "delta": None}
        with pytest.raises(ReviewValidationError, match="primary_metric.*name"):
            validate_review_decision(record)

    def test_invalid_risk_flag_type(self):
        record = _make_record(risk_flags_summary=[
            {"type": "unknown_type", "severity": "low", "evidence": "test"}
        ])
        with pytest.raises(ReviewValidationError, match="risk_flags_summary"):
            validate_review_decision(record)

    def test_valid_record_with_risk_flags(self):
        record = _make_record(risk_flags_summary=[
            {"type": "overfitting", "severity": "medium", "evidence": "Gap > 10%"},
            {"type": "leakage", "severity": "low", "evidence": "Slightly high metric"},
        ])
        summary = validate_review_decision(record)
        assert summary["risk_flag_count"] == 2

    def test_from_json_file(self, tmp_path):
        record = _make_record()
        path = tmp_path / "record.json"
        path.write_text(json.dumps(record))
        summary = validate_review_decision(path)
        assert summary["iteration"] == 1


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------

class TestHistory:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "run-history.jsonl"
        path.write_text("")
        records = load_run_history(path)
        assert records == []

    def test_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        records = load_run_history(path)
        assert records == []

    def test_load_records(self, tmp_path):
        path = tmp_path / "run-history.jsonl"
        lines = [json.dumps(_make_record(i)) for i in [2, 1, 3]]
        path.write_text("\n".join(lines) + "\n")
        records = load_run_history(path)
        assert len(records) == 3
        assert records[0]["iteration"] == 1  # sorted
        assert records[2]["iteration"] == 3

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "run-history.jsonl"
        path.write_text('{"valid": 1}\nnot json\n')
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_run_history(path)

    def test_summarise_empty(self):
        summary = summarise_history([])
        assert summary["total_iterations"] == 0
        assert summary["best_iteration"] is None

    def test_summarise_records(self):
        records = [_make_record(i) for i in [1, 2, 3]]
        records[1]["model_family"] = "random_forest"
        records[2]["risk_flags_summary"] = [
            {"type": "overfitting", "severity": "high", "evidence": "test"}
        ]
        summary = summarise_history(records)
        assert summary["total_iterations"] == 3
        assert summary["best_iteration"] == 3  # highest value
        assert len(summary["model_families_tried"]) == 2
        assert summary["has_high_severity_flags"] is True


# ---------------------------------------------------------------------------
# Comparator tests
# ---------------------------------------------------------------------------

class TestComparator:
    def test_deltas_first_iteration(self):
        result = compute_deltas(0.83, None, None)
        assert result["delta_vs_previous"] is None
        assert result["delta_vs_best"] is None

    def test_deltas_improvement(self):
        result = compute_deltas(0.85, 0.83, 0.83)
        assert result["delta_vs_previous"] == pytest.approx(0.02)
        assert result["improved_vs_best"] is True

    def test_deltas_degradation(self):
        result = compute_deltas(0.80, 0.83, 0.85)
        assert result["delta_vs_previous"] == pytest.approx(-0.03)
        assert result["improved_vs_best"] is False

    def test_trend_insufficient_data(self):
        trajectory = [{"iteration": 1, "value": 0.80, "delta": None}]
        assert compute_trend(trajectory) == "insufficient_data"

    def test_trend_improving(self):
        trajectory = [
            {"iteration": 1, "value": 0.80, "delta": None},
            {"iteration": 2, "value": 0.83, "delta": 0.03},
            {"iteration": 3, "value": 0.86, "delta": 0.03},
        ]
        assert compute_trend(trajectory) == "improving"

    def test_trend_plateau(self):
        trajectory = [
            {"iteration": 1, "value": 0.80, "delta": None},
            {"iteration": 2, "value": 0.801, "delta": 0.001},
            {"iteration": 3, "value": 0.802, "delta": 0.001},
        ]
        assert compute_trend(trajectory) == "plateau"

    def test_trend_degrading(self):
        trajectory = [
            {"iteration": 1, "value": 0.85, "delta": None},
            {"iteration": 2, "value": 0.83, "delta": -0.02},
            {"iteration": 3, "value": 0.80, "delta": -0.03},
        ]
        assert compute_trend(trajectory) == "degrading"

    def test_find_best_empty(self):
        result = find_best_iteration([])
        assert result["best_iteration"] is None

    def test_find_best(self):
        records = [_make_record(i) for i in [1, 2, 3]]
        records[1]["primary_metric"]["value"] = 0.95
        result = find_best_iteration(records)
        assert result["best_iteration"] == 2
        assert result["best_value"] == 0.95


# ---------------------------------------------------------------------------
# Plateau tests
# ---------------------------------------------------------------------------

class TestPlateau:
    def test_no_plateau_first_iteration(self):
        report = {"reviewer_summary": {"plateau_signal": {"detected": False, "consecutive_stale_iterations": 0}}}
        result = check_plateau(report, [])
        assert result["detected"] is False

    def test_plateau_from_model_report(self):
        report = {"reviewer_summary": {"plateau_signal": {"detected": True, "consecutive_stale_iterations": 4}}}
        records = [_make_record(i, model_family=f"family_{i % 2}") for i in [1, 2, 3]]
        result = check_plateau(report, records)
        assert result["detected"] is True
        assert result["source"] == "model_report"
        assert result["multiple_strategies_tried"] is True

    def test_plateau_from_history(self):
        report = {"reviewer_summary": {"plateau_signal": {"detected": False, "consecutive_stale_iterations": 0}}}
        records = [
            _make_record(i, primary_metric={"name": "auc", "value": 0.80, "delta": 0.001})
            for i in [1, 2, 3]
        ]
        result = check_plateau(report, records)
        assert result["detected"] is True
        assert result["source"] == "history_computed"

    def test_no_plateau_with_improvement(self):
        report = {"reviewer_summary": {"plateau_signal": {"detected": False, "consecutive_stale_iterations": 0}}}
        records = [
            _make_record(i, primary_metric={"name": "auc", "value": 0.80 + i * 0.05, "delta": 0.05})
            for i in [1, 2, 3]
        ]
        result = check_plateau(report, records)
        assert result["detected"] is False

    def test_single_strategy(self):
        report = {"reviewer_summary": {"plateau_signal": {"detected": True, "consecutive_stale_iterations": 3}}}
        records = [_make_record(i) for i in [1, 2, 3]]  # all same model_family
        result = check_plateau(report, records)
        assert result["multiple_strategies_tried"] is False


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------

class TestWriter:
    def test_build_record(self):
        record = build_record(
            iteration=1,
            status="completed",
            plan_summary="Baseline LR",
            primary_metric_name="auc_roc",
            primary_metric_value=0.835,
            primary_metric_delta=None,
            model_family="logistic_regression",
            reviewer_verdict="insufficient",
            reviewer_reasoning="Room for improvement",
            router_decision="continue",
            router_reasoning="First iteration baseline",
            risk_flags_summary=[],
            best_iteration=1,
        )
        assert record["iteration"] == 1
        assert record["timestamp"]  # auto-generated
        assert record["primary_metric"]["name"] == "auc_roc"

    def test_append_creates_file(self, tmp_path):
        record = _make_record()
        path = tmp_path / "memory" / "run-history.jsonl"
        summary = append_review_decision(record, path)
        assert path.exists()
        assert summary["iteration"] == 1

        # Verify JSONL content
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["reviewer_verdict"] == "insufficient"

    def test_append_multiple(self, tmp_path):
        path = tmp_path / "run-history.jsonl"
        for i in [1, 2, 3]:
            append_review_decision(_make_record(i), path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_append_writes_per_iteration_json(self, tmp_path):
        record = _make_record()
        history_path = tmp_path / "memory" / "run-history.jsonl"
        iter_dir = tmp_path / "iterations" / "iteration-1"
        iter_dir.mkdir(parents=True)

        append_review_decision(record, history_path, iteration_dir=iter_dir)

        review_path = iter_dir / "reports" / "review-decision.json"
        assert review_path.exists()
        parsed = json.loads(review_path.read_text())
        assert parsed["iteration"] == 1
        assert parsed["reviewer_verdict"] == "insufficient"

    def test_append_without_iteration_dir(self, tmp_path):
        record = _make_record()
        history_path = tmp_path / "run-history.jsonl"
        append_review_decision(record, history_path)
        # No iteration dir → no review-decision.json written, just JSONL
        assert history_path.exists()

    def test_append_invalid_record(self, tmp_path):
        path = tmp_path / "run-history.jsonl"
        record = _make_record()
        del record["reviewer_verdict"]
        with pytest.raises(ReviewValidationError):
            append_review_decision(record, path)
        # File should NOT have been written
        assert not path.exists()
