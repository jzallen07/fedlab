from __future__ import annotations

import pytest

from src.monitor.control import InvalidTransitionError, next_state


def test_valid_transition_running_to_paused() -> None:
    assert next_state("running", "pause") == "paused"


def test_invalid_transition_idle_to_resume_raises() -> None:
    with pytest.raises(InvalidTransitionError):
        next_state("idle", "resume")
