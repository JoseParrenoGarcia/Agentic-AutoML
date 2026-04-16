"""Review configuration defaults.

Future orchestrator will override these via experiment config.
"""

MAX_ITERATIONS = 10
"""Hard stop: router must emit 'sufficient' when iteration >= MAX_ITERATIONS."""

PLATEAU_STALE_THRESHOLD = 0.005
"""Absolute delta below which an iteration is considered stale."""

PLATEAU_CONSECUTIVE_MIN = 3
"""Number of consecutive stale iterations before plateau is confirmed."""
