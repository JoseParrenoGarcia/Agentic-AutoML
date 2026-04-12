from pathlib import Path

import pytest

from src.planning.validator import PlanValidationError, validate_plan

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_valid_plan_passes():
    """Load valid_plan.yaml, assert no exception, return value is dict."""
    result = validate_plan(FIXTURES / "valid_plan.yaml")
    assert isinstance(result, dict)
    assert result["iteration"] == 1


def test_valid_plan_as_dict_passes():
    """Pass a pre-parsed dict directly, assert passes."""
    plan = {
        "iteration": 1,
        "objective": "Establish baseline.",
        "hypotheses": [{"id": "H1", "description": "Test.", "expected_impact": "Moderate."}],
        "feature_steps": [],
        "model_steps": [{"algorithm": "LogisticRegression", "hyperparameters": {}, "rationale": "Interpretable."}],
        "evaluation_focus": "AUC-ROC.",
        "expected_win_condition": "AUC-ROC > 0.80.",
        "rollback_or_stop_condition": "AUC-ROC < 0.70.",
    }
    result = validate_plan(plan)
    assert isinstance(result, dict)


def test_missing_objective_raises():
    """PlanValidationError with message mentioning 'objective'."""
    with pytest.raises(PlanValidationError, match="objective"):
        validate_plan(FIXTURES / "invalid_plan_missing_objective.yaml")


def test_missing_hypotheses_raises():
    """PlanValidationError with message mentioning 'hypotheses'."""
    plan = {
        "iteration": 1,
        "objective": "Establish baseline.",
        "feature_steps": [],
        "model_steps": [{"algorithm": "LR", "hyperparameters": {}, "rationale": "OK."}],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    with pytest.raises(PlanValidationError, match="hypotheses"):
        validate_plan(plan)


def test_missing_model_steps_raises():
    """PlanValidationError with message mentioning 'model_steps'."""
    plan = {
        "iteration": 1,
        "objective": "Establish baseline.",
        "hypotheses": [{"id": "H1", "description": "Test.", "expected_impact": "Moderate."}],
        "feature_steps": [],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    with pytest.raises(PlanValidationError, match="model_steps"):
        validate_plan(plan)


def test_empty_hypotheses_list_raises():
    """hypotheses = [] should raise PlanValidationError."""
    plan = {
        "iteration": 1,
        "objective": "Establish baseline.",
        "hypotheses": [],
        "feature_steps": [],
        "model_steps": [{"algorithm": "LR", "hyperparameters": {}, "rationale": "OK."}],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    with pytest.raises(PlanValidationError, match="hypotheses"):
        validate_plan(plan)


def test_empty_model_steps_list_raises():
    """model_steps = [] should raise PlanValidationError."""
    plan = {
        "iteration": 1,
        "objective": "Establish baseline.",
        "hypotheses": [{"id": "H1", "description": "Test.", "expected_impact": "Moderate."}],
        "feature_steps": [],
        "model_steps": [],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    with pytest.raises(PlanValidationError, match="model_steps"):
        validate_plan(plan)


def test_feature_steps_may_be_empty():
    """feature_steps = [] is valid — should NOT raise."""
    plan = {
        "iteration": 1,
        "objective": "Model-only change.",
        "hypotheses": [{"id": "H1", "description": "Test.", "expected_impact": "Moderate."}],
        "feature_steps": [],
        "model_steps": [{"algorithm": "LR", "hyperparameters": {}, "rationale": "OK."}],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    result = validate_plan(plan)
    assert result["feature_steps"] == []


def test_invalid_iteration_type_raises():
    """iteration = 'first' (string) should raise PlanValidationError."""
    plan = {
        "iteration": "first",
        "objective": "Establish baseline.",
        "hypotheses": [{"id": "H1", "description": "Test.", "expected_impact": "Moderate."}],
        "feature_steps": [],
        "model_steps": [{"algorithm": "LR", "hyperparameters": {}, "rationale": "OK."}],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    with pytest.raises(PlanValidationError, match="iteration"):
        validate_plan(plan)


def test_iteration_zero_raises():
    """iteration = 0 should raise (must be >= 1)."""
    plan = {
        "iteration": 0,
        "objective": "Establish baseline.",
        "hypotheses": [{"id": "H1", "description": "Test.", "expected_impact": "Moderate."}],
        "feature_steps": [],
        "model_steps": [{"algorithm": "LR", "hyperparameters": {}, "rationale": "OK."}],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    with pytest.raises(PlanValidationError, match="iteration"):
        validate_plan(plan)


def test_missing_sub_field_in_hypothesis_raises():
    """Hypothesis with no 'expected_impact' should raise PlanValidationError."""
    plan = {
        "iteration": 1,
        "objective": "Establish baseline.",
        "hypotheses": [{"id": "H1", "description": "Test."}],
        "feature_steps": [],
        "model_steps": [{"algorithm": "LR", "hyperparameters": {}, "rationale": "OK."}],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    with pytest.raises(PlanValidationError, match="expected_impact"):
        validate_plan(plan)


def test_missing_rationale_in_model_step_raises():
    """Model step with no 'rationale' should raise PlanValidationError."""
    plan = {
        "iteration": 1,
        "objective": "Establish baseline.",
        "hypotheses": [{"id": "H1", "description": "Test.", "expected_impact": "Moderate."}],
        "feature_steps": [],
        "model_steps": [{"algorithm": "LR", "hyperparameters": {}}],
        "evaluation_focus": "AUC.",
        "expected_win_condition": "AUC > 0.8.",
        "rollback_or_stop_condition": "AUC < 0.7.",
    }
    with pytest.raises(PlanValidationError, match="rationale"):
        validate_plan(plan)


def test_nonexistent_file_raises_file_not_found():
    """Passing a path that doesn't exist should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        validate_plan(FIXTURES / "does_not_exist.yaml")
