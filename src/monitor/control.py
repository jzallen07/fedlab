"""Run-state transition policy for monitor control endpoints."""

from __future__ import annotations

from src.monitor.schema import RunAction, RunState

_TRANSITIONS: dict[RunState, dict[RunAction, RunState]] = {
    "idle": {"start": "running"},
    "running": {"pause": "paused", "stop": "stopped"},
    "paused": {"resume": "running", "stop": "stopped"},
    "stopped": {"start": "running"},
    "error": {"stop": "stopped", "start": "running"},
}


class InvalidTransitionError(ValueError):
    """Raised when a run-state transition is invalid."""


def next_state(current: RunState, action: RunAction) -> RunState:
    """Resolve the next state for a requested control action."""
    actions = _TRANSITIONS.get(current, {})
    if action not in actions:
        raise InvalidTransitionError(f"action '{action}' invalid for state '{current}'")
    return actions[action]
