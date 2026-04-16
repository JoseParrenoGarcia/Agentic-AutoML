# %% [utils] Shared utilities — logging, timing, warnings, seed, metadata
# Used by all other modules. Import at the top of main.py only; other modules
# receive logger/metadata objects as arguments where needed.

import logging
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    """
    Create a logger that writes to both stdout and a file.

    Args:
        log_path: Path to the log file (sourced from config output_paths.log).

    Returns:
        Configured Logger instance.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("iteration")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed: Integer seed value (sourced from config.random_seed).
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(stage_name: str, metadata: "PipelineMetadata") -> Generator:
    """
    Context manager that measures wall-clock duration of a pipeline stage and
    records it in the PipelineMetadata collector.

    Usage:
        with timer("feature_engineering", metadata):
            df = engineer_features(df, config)
    """
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    metadata.record_duration(stage_name, duration)


# ---------------------------------------------------------------------------
# Warning capture
# ---------------------------------------------------------------------------

@contextmanager
def capture_warnings(stage_name: str, metadata: "PipelineMetadata") -> Generator:
    """
    Context manager that captures Python warnings emitted during a stage and
    stores them in the PipelineMetadata collector.

    Usage:
        with capture_warnings("model_training", metadata):
            model = train_model(X_train, y_train, config)
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        metadata.record_warning(stage_name, str(w.message))


# ---------------------------------------------------------------------------
# Pipeline metadata collector
# ---------------------------------------------------------------------------

class PipelineMetadata:
    """
    Accumulates per-stage shape, duration, and warning information during a run.
    Written to outputs/pipeline_metadata.json by evaluate.py at the end of the run.

    Schema written:
        {
            "stages": [
                {
                    "name": str,
                    "input_shape": [int, int],
                    "output_shape": [int, int],
                    "duration_s": float,
                    "warnings": [str]
                },
                ...
            ],
            "total_duration_s": float,
            "python_version": str,
            "packages": {name: version, ...}
        }
    """

    def __init__(self) -> None:
        self._stages: list[dict] = []
        self._durations: dict[str, float] = {}
        self._warnings: dict[str, list[str]] = {}

    def record_stage(
        self,
        name: str,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
    ) -> None:
        """Record shape information for a pipeline stage."""
        self._stages.append({
            "name": name,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "duration_s": self._durations.get(name, 0.0),
            "warnings": self._warnings.get(name, []),
        })

    def record_duration(self, stage_name: str, duration_s: float) -> None:
        self._durations[stage_name] = round(duration_s, 4)

    def record_warning(self, stage_name: str, message: str) -> None:
        self._warnings.setdefault(stage_name, []).append(message)

    def to_dict(self) -> dict:
        """Serialise to the pipeline_metadata.json schema."""
        import sys
        import importlib.metadata

        packages = {}
        for pkg in ("scikit-learn", "pandas", "numpy"):
            try:
                packages[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                packages[pkg] = "unknown"

        return {
            "stages": self._stages,
            "total_duration_s": round(sum(self._durations.values()), 4),
            "python_version": sys.version,
            "packages": packages,
        }
