"""Unit tests for M8 — Project Memory (decision-log writer + retrieval)."""

import json
import re
from pathlib import Path

import pytest

from src.review.writer import (
    append_decision_log,
    append_review_decision,
    build_record,
    format_decision_log_entry,
)
from src.review.history import load_run_history, summarise_history
from src.review.validator import ReviewValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(iteration=1, verdict="insufficient", route="continue", **overrides):
    """Helper to build a valid review-decision dict."""
    base = {
        "iteration": iteration,
        "timestamp": "2026-04-16T10:00:00+00:00",
        "status": "completed",
        "plan_summary": f"Iteration {iteration} plan summary",
        "primary_metric": {
            "name": "val_auc_roc",
            "value": 0.80 + iteration * 0.01,
            "delta": None if iteration == 1 else 0.01,
        },
        "model_family": "LogisticRegression" if iteration == 1 else "GradientBoosting",
        "reviewer_verdict": verdict,
        "reviewer_reasoning": f"Reasoning for iteration {iteration} with enough detail to test truncation behavior.",
        "router_decision": route,
        "router_reasoning": f"Route reasoning for iteration {iteration}",
        "risk_flags_summary": [],
        "best_iteration": iteration,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test 1: append_decision_log creates file if missing
# ---------------------------------------------------------------------------

class TestDecisionLogCreation:
    def test_creates_file_if_missing(self, tmp_path):
        log_path = tmp_path / "memory" / "decision-log.md"
        assert not log_path.exists()

        append_decision_log(_make_record(1), log_path)

        assert log_path.exists()
        content = log_path.read_text()
        assert "# Decision Log" in content
        assert "## Iteration 1" in content


# ---------------------------------------------------------------------------
# Test 2: append_decision_log appends without overwriting
# ---------------------------------------------------------------------------

class TestDecisionLogAppend:
    def test_appends_without_overwriting(self, tmp_path):
        log_path = tmp_path / "decision-log.md"

        append_decision_log(_make_record(1), log_path)
        append_decision_log(_make_record(2, route="rollback"), log_path)

        content = log_path.read_text()
        assert content.count("## Iteration") == 2
        assert "## Iteration 1" in content
        assert "## Iteration 2" in content


# ---------------------------------------------------------------------------
# Test 3: Decision-log entry format matches expected structure
# ---------------------------------------------------------------------------

class TestDecisionLogFormat:
    def test_entry_has_required_fields(self):
        record = _make_record(1)
        entry = format_decision_log_entry(record)

        assert "## Iteration 1 — LogisticRegression" in entry
        assert "**Metric:**" in entry
        assert "val_auc_roc" in entry
        assert "**Verdict:** insufficient" in entry
        assert "**Route:** continue" in entry
        assert "**Summary:**" in entry
        assert "**Reasoning:**" in entry
        assert "**Risk flags:**" in entry

    def test_baseline_delta_shows_na(self):
        record = _make_record(1)
        entry = format_decision_log_entry(record)
        assert "N/A (baseline)" in entry

    def test_nonbaseline_delta_shows_value(self):
        record = _make_record(2, route="rollback")
        record["primary_metric"]["delta"] = -0.013
        entry = format_decision_log_entry(record)
        assert "-0.0130" in entry

    def test_risk_flags_formatted(self):
        record = _make_record(2)
        record["risk_flags_summary"] = [
            {"type": "overfitting", "severity": "high", "evidence": "gap 17%"}
        ]
        entry = format_decision_log_entry(record)
        assert "1 (high-overfitting)" in entry

    def test_long_reasoning_truncated(self):
        record = _make_record(1)
        record["reviewer_reasoning"] = "x" * 300
        entry = format_decision_log_entry(record)
        assert "..." in entry
        # Should have 200 chars + "..."
        reasoning_line = [l for l in entry.split("\n") if "**Reasoning:**" in l][0]
        # The truncated reasoning should be present but not the full 300 chars
        assert "x" * 200 in reasoning_line
        assert "x" * 201 not in reasoning_line


# ---------------------------------------------------------------------------
# Test 4: Consistency — JSONL count == decision-log heading count
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_jsonl_and_decision_log_stay_in_sync(self, tmp_path):
        history_path = tmp_path / "run-history.jsonl"
        log_path = tmp_path / "decision-log.md"

        for i in range(1, 6):
            record = _make_record(i)
            append_review_decision(
                record, history_path, decision_log_path=log_path
            )

        # Count JSONL lines
        jsonl_lines = [l for l in history_path.read_text().strip().split("\n") if l.strip()]
        # Count markdown headings
        log_content = log_path.read_text()
        heading_count = log_content.count("## Iteration")

        assert len(jsonl_lines) == 5
        assert heading_count == 5


# ---------------------------------------------------------------------------
# Test 5: Validation failure does NOT write to decision-log (atomicity)
# ---------------------------------------------------------------------------

class TestAtomicity:
    def test_validation_failure_skips_decision_log(self, tmp_path):
        history_path = tmp_path / "run-history.jsonl"
        log_path = tmp_path / "decision-log.md"

        # Write one valid record first
        append_review_decision(
            _make_record(1), history_path, decision_log_path=log_path
        )
        assert log_path.read_text().count("## Iteration") == 1

        # Attempt an invalid record (missing required field)
        bad_record = {"iteration": 2, "status": "completed"}
        with pytest.raises(ReviewValidationError):
            append_review_decision(
                bad_record, history_path, decision_log_path=log_path
            )

        # Decision log should still have only 1 entry
        assert log_path.read_text().count("## Iteration") == 1
        # JSONL should still have only 1 line
        jsonl_lines = [l for l in history_path.read_text().strip().split("\n") if l.strip()]
        assert len(jsonl_lines) == 1


# ---------------------------------------------------------------------------
# Test 6: summarise_history returns correct best_iteration across 3+ records
# ---------------------------------------------------------------------------

class TestSummariseHistory:
    def test_best_iteration_across_multiple(self, tmp_path):
        history_path = tmp_path / "run-history.jsonl"

        records = [
            _make_record(1, primary_metric={"name": "auc", "value": 0.83, "delta": None}),
            _make_record(2, primary_metric={"name": "auc", "value": 0.79, "delta": -0.04}),
            _make_record(3, primary_metric={"name": "auc", "value": 0.86, "delta": 0.07}),
        ]
        for r in records:
            history_path.open("a").write(json.dumps(r) + "\n")

        loaded = load_run_history(history_path)
        summary = summarise_history(loaded)

        assert summary["best_iteration"] == 3
        assert summary["best_metric_value"] == 0.86
        assert summary["total_iterations"] == 3

    def test_best_iteration_when_later_regresses(self, tmp_path):
        history_path = tmp_path / "run-history.jsonl"

        records = [
            _make_record(1, primary_metric={"name": "auc", "value": 0.85, "delta": None}),
            _make_record(2, primary_metric={"name": "auc", "value": 0.82, "delta": -0.03}),
            _make_record(3, primary_metric={"name": "auc", "value": 0.84, "delta": 0.02}),
        ]
        for r in records:
            history_path.open("a").write(json.dumps(r) + "\n")

        loaded = load_run_history(history_path)
        summary = summarise_history(loaded)

        assert summary["best_iteration"] == 1
        assert summary["best_metric_value"] == 0.85


# ---------------------------------------------------------------------------
# Test 7: summarise_history handles empty history
# ---------------------------------------------------------------------------

class TestSummariseEmpty:
    def test_empty_history(self):
        summary = summarise_history([])
        assert summary["total_iterations"] == 0
        assert summary["best_iteration"] is None
        assert summary["model_families_tried"] == []
        assert summary["metric_trajectory"] == []

    def test_missing_file(self, tmp_path):
        loaded = load_run_history(tmp_path / "nonexistent.jsonl")
        assert loaded == []


# ---------------------------------------------------------------------------
# Test 8: Backfill produces entries matching live-written entries
# ---------------------------------------------------------------------------

class TestBackfillConsistency:
    def test_backfill_matches_live_write(self, tmp_path):
        record = _make_record(1)

        # Simulate live write (via append_review_decision)
        live_log = tmp_path / "live" / "decision-log.md"
        append_review_decision(
            record,
            tmp_path / "live" / "run-history.jsonl",
            decision_log_path=live_log,
        )

        # Simulate backfill (direct append_decision_log)
        backfill_log = tmp_path / "backfill" / "decision-log.md"
        append_decision_log(record, backfill_log)

        # Extract just the iteration entry (skip header which may differ slightly)
        live_entry = re.search(r"## Iteration.*", live_log.read_text(), re.DOTALL)
        backfill_entry = re.search(r"## Iteration.*", backfill_log.read_text(), re.DOTALL)

        assert live_entry is not None
        assert backfill_entry is not None
        assert live_entry.group() == backfill_entry.group()


# ---------------------------------------------------------------------------
# Test 9: Round-trip write → load → summarise
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_write_load_summarise(self, tmp_path):
        history_path = tmp_path / "run-history.jsonl"
        log_path = tmp_path / "decision-log.md"

        families = ["LogisticRegression", "GradientBoosting", "RandomForest"]
        for i in range(1, 4):
            record = _make_record(
                i,
                model_family=families[i - 1],
                primary_metric={"name": "auc", "value": 0.80 + i * 0.02, "delta": 0.02 if i > 1 else None},
            )
            append_review_decision(record, history_path, decision_log_path=log_path)

        loaded = load_run_history(history_path)
        summary = summarise_history(loaded)

        assert summary["total_iterations"] == 3
        assert summary["model_families_tried"] == families
        assert len(summary["metric_trajectory"]) == 3
        assert summary["best_metric_value"] == pytest.approx(0.86)

        log_content = log_path.read_text()
        for fam in families:
            assert fam in log_content


# ---------------------------------------------------------------------------
# Test 10: Existing decision-log.md content is preserved on append
# ---------------------------------------------------------------------------

class TestPreservesExistingContent:
    def test_preserves_prior_content(self, tmp_path):
        log_path = tmp_path / "decision-log.md"

        # Write first entry
        append_decision_log(_make_record(1), log_path)
        first_content = log_path.read_text()

        # Write second entry
        append_decision_log(_make_record(2), log_path)
        second_content = log_path.read_text()

        # First entry's content should still be present
        assert "## Iteration 1" in second_content
        assert "## Iteration 2" in second_content
        # The header should appear only once
        assert second_content.count("# Decision Log") == 1
