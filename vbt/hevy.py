"""Backwards-compatible entry point; see vbt.logs for every supported log format."""
from .logs import parse_hevy


def parse_hevy_text(text):
    """Hevy share text -> list of sets (exercise, kg, reps, rpe, warmup, set)."""
    return parse_hevy(text)[0]
