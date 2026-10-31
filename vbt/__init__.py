"""Velocity-based training and rep counting from Apple Watch motion data."""
from .io import load_watch_csv
from .core import analyze, vertical_acceleration, detect_reps
from .loadvelocity import fit_load_velocity, estimate_1rm
from .synthetic import synth_set

__all__ = ["load_watch_csv", "analyze", "vertical_acceleration", "detect_reps",
           "fit_load_velocity", "estimate_1rm", "synth_set"]
