"""Stroke detector package for thin hand-drawn curves on textured cardboard."""

from .config import Config
from .pipeline import detect_strokes, run_pipeline

__all__ = ["Config", "detect_strokes", "run_pipeline"]

