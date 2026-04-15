import pytest

from src.execution.classifier import (
    ErrorCategory,
    ErrorClassification,
    Stage,
    classify_error,
)


@pytest.mark.parametrize(
    "stderr, exit_code, expected_category, expected_stage",
    [
        pytest.param(
            'Traceback (most recent call last):\n  File "src/main.py", line 10\n    print(\nSyntaxError: unexpected EOF while parsing',
            1,
            ErrorCategory.SYNTAX_ERROR,
            Stage.STAGE_1,
            id="syntax_error",
        ),
        pytest.param(
            "Traceback (most recent call last):\n  File \"src/model.py\", line 3, in <module>\n    import xgboost\nModuleNotFoundError: No module named 'xgboost'",
            1,
            ErrorCategory.IMPORT_ERROR,
            Stage.STAGE_1,
            id="import_error",
        ),
        pytest.param(
            'Traceback (most recent call last):\n  File "src/main.py", line 20, in <module>\nTypeError: cannot unpack non-iterable NoneType object',
            1,
            ErrorCategory.TYPE_ERROR,
            Stage.STAGE_1,
            id="type_error",
        ),
        pytest.param(
            'Traceback (most recent call last):\n  File "src/model.py", line 42, in train_model\nValueError: shapes (100,5) and (3,) not aligned',
            1,
            ErrorCategory.DATA_SHAPE_ERROR,
            Stage.STAGE_2,
            id="data_shape_error",
        ),
        pytest.param(
            'Traceback (most recent call last):\n  File "src/feature_engineering.py", line 15\nValueError: Input X contains NaN',
            1,
            ErrorCategory.NAN_PROPAGATION,
            Stage.STAGE_2,
            id="nan_propagation",
        ),
        pytest.param(
            'Traceback (most recent call last):\n  File "src/model.py", line 30\nConvergenceWarning: lbfgs failed to converge (status=1)',
            1,
            ErrorCategory.CONVERGENCE_ERROR,
            Stage.STAGE_2,
            id="convergence_error",
        ),
        pytest.param(
            "",
            -9,
            ErrorCategory.TIMEOUT,
            Stage.STAGE_2,
            id="timeout",
        ),
        pytest.param(
            "Traceback (most recent call last):\n  File \"src/data_loader.py\", line 8\nKeyError: 'missing_column'",
            1,
            ErrorCategory.RUNTIME_ERROR,
            Stage.STAGE_2,
            id="runtime_error",
        ),
        pytest.param(
            "",
            1,
            ErrorCategory.UNKNOWN,
            Stage.STAGE_2,
            id="unknown_no_stderr",
        ),
    ],
)
def test_classify_error(stderr, exit_code, expected_category, expected_stage):
    result = classify_error(stderr, exit_code)
    assert isinstance(result, ErrorClassification)
    assert result.category == expected_category
    assert result.stage == expected_stage


def test_traceback_extraction():
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "src/main.py", line 5, in <module>\n'
        '    from model import train_model\n'
        '  File "src/model.py", line 42, in <module>\n'
        '    raise RuntimeError("boom")\n'
        'RuntimeError: boom'
    )
    result = classify_error(stderr, 1)
    assert result.file_hint == "model.py"
    assert result.line_hint == 42


def test_traceback_prefers_user_code_over_library():
    """file_hint should point to src/ code, not library internals."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "src/feature_engineering.py", line 112, in fit_transform\n'
        '    df["AgeXYZ"]\n'
        '  File "/usr/lib/python3.11/site-packages/pandas/core/frame.py", line 3648, in __getitem__\n'
        '    indexer = self.columns.get_loc(key)\n'
        'KeyError: \'AgeXYZ\''
    )
    result = classify_error(stderr, 1)
    assert result.file_hint == "feature_engineering.py"
    assert result.line_hint == 112
