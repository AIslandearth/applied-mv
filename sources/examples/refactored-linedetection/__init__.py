"""line_detector — modular Hough-line detection package."""

from .geometry import to_slope_intercept, line_intersections
from .detector import LineDetector
from .roi import ROISelector
from .app import App

__all__ = [
    "to_slope_intercept",
    "line_intersections",
    "LineDetector",
    "ROISelector",
    "App",
]
