import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    DATA_SHAPE_ERROR = "data_shape_error"
    NAN_PROPAGATION = "nan_propagation"
    CONVERGENCE_ERROR = "convergence_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class Stage(int, Enum):
    STAGE_1 = 1  # Syntax / Import — max 3 retries
    STAGE_2 = 2  # Logic / Runtime — max 2 retries


STAGE_MAP: dict[ErrorCategory, Stage] = {
    ErrorCategory.SYNTAX_ERROR: Stage.STAGE_1,
    ErrorCategory.IMPORT_ERROR: Stage.STAGE_1,
    ErrorCategory.TYPE_ERROR: Stage.STAGE_1,
    ErrorCategory.DATA_SHAPE_ERROR: Stage.STAGE_2,
    ErrorCategory.NAN_PROPAGATION: Stage.STAGE_2,
    ErrorCategory.CONVERGENCE_ERROR: Stage.STAGE_2,
    ErrorCategory.RUNTIME_ERROR: Stage.STAGE_2,
    ErrorCategory.TIMEOUT: Stage.STAGE_2,
    ErrorCategory.UNKNOWN: Stage.STAGE_2,
}

STAGE_LIMITS: dict[Stage, int] = {
    Stage.STAGE_1: 3,
    Stage.STAGE_2: 2,
}

TOTAL_ATTEMPT_CAP = 5


@dataclass
class ErrorClassification:
    category: ErrorCategory
    stage: Stage
    summary: str
    file_hint: Optional[str] = None
    line_hint: Optional[int] = None


# Regex for extracting file and line from Python tracebacks:
#   File "src/model.py", line 42, in train_model
_TB_FILE_RE = re.compile(r'File "([^"]+)", line (\d+)')


def _extract_location(stderr: str) -> tuple[Optional[str], Optional[int]]:
    """Extract file_hint and line_hint from the last user-code traceback frame.

    Prefers frames within ``src/`` over library frames. Falls back to the
    last frame if no user-code frame is found.
    """
    matches = _TB_FILE_RE.findall(stderr)
    if not matches:
        return None, None

    # Prefer the last frame that looks like user code (contains "src/")
    best = matches[-1]
    for file_path, line_str in reversed(matches):
        if "src/" in file_path:
            best = (file_path, line_str)
            break

    file_path, line_str = best
    filename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
    return filename, int(line_str)


def _make_summary(stderr: str, max_len: int = 500) -> str:
    """Extract a concise error summary from stderr."""
    lines = stderr.strip().splitlines()
    # Try to find the actual error line (last line starting with an error type)
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("File ") and not stripped.startswith("^"):
            return stripped[:max_len]
    return stderr[:max_len].strip()


def classify_error(stderr: str, exit_code: int) -> ErrorClassification:
    """
    Classify a runtime error from captured stderr and exit code.

    Rules are applied in order — first match wins. The classification
    tells the executor agent *what* failed so it can decide *how* to fix it.

    Args:
        stderr: Captured stderr from the subprocess.
        exit_code: Process exit code (-9 indicates timeout).

    Returns:
        ErrorClassification with category, stage, summary, and location hints.
    """
    file_hint, line_hint = _extract_location(stderr)
    summary = _make_summary(stderr)

    # Rule 1: Timeout
    if exit_code == -9:
        return ErrorClassification(
            category=ErrorCategory.TIMEOUT,
            stage=Stage.STAGE_2,
            summary=summary or "Process killed: timeout exceeded",
            file_hint=file_hint,
            line_hint=line_hint,
        )

    # Rule 2: SyntaxError
    if "SyntaxError" in stderr:
        return ErrorClassification(
            category=ErrorCategory.SYNTAX_ERROR,
            stage=Stage.STAGE_1,
            summary=summary,
            file_hint=file_hint,
            line_hint=line_hint,
        )

    # Rule 3: Import / Module errors
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        return ErrorClassification(
            category=ErrorCategory.IMPORT_ERROR,
            stage=Stage.STAGE_1,
            summary=summary,
            file_hint=file_hint,
            line_hint=line_hint,
        )

    # Rule 4: TypeError
    if "TypeError" in stderr:
        return ErrorClassification(
            category=ErrorCategory.TYPE_ERROR,
            stage=Stage.STAGE_1,
            summary=summary,
            file_hint=file_hint,
            line_hint=line_hint,
        )

    # Rule 5: Data shape / dimension errors
    shape_keywords = ("shape", "dimension", "mismatch", "broadcast", "aligned")
    stderr_lower = stderr.lower()
    if any(kw in stderr_lower for kw in shape_keywords):
        return ErrorClassification(
            category=ErrorCategory.DATA_SHAPE_ERROR,
            stage=Stage.STAGE_2,
            summary=summary,
            file_hint=file_hint,
            line_hint=line_hint,
        )

    # Rule 6: NaN / inf propagation
    if re.search(r"(nan|NaN|inf)\b", stderr) and "ValueError" in stderr:
        return ErrorClassification(
            category=ErrorCategory.NAN_PROPAGATION,
            stage=Stage.STAGE_2,
            summary=summary,
            file_hint=file_hint,
            line_hint=line_hint,
        )

    # Rule 7: Convergence failures
    if "ConvergenceWarning" in stderr or "did not converge" in stderr.lower():
        return ErrorClassification(
            category=ErrorCategory.CONVERGENCE_ERROR,
            stage=Stage.STAGE_2,
            summary=summary,
            file_hint=file_hint,
            line_hint=line_hint,
        )

    # Rule 8: Any other traceback → RUNTIME_ERROR
    if "Traceback (most recent call last)" in stderr:
        return ErrorClassification(
            category=ErrorCategory.RUNTIME_ERROR,
            stage=Stage.STAGE_2,
            summary=summary,
            file_hint=file_hint,
            line_hint=line_hint,
        )

    # Rule 9: No traceback → UNKNOWN
    return ErrorClassification(
        category=ErrorCategory.UNKNOWN,
        stage=Stage.STAGE_2,
        summary=summary or f"Process exited with code {exit_code}",
        file_hint=file_hint,
        line_hint=line_hint,
    )
