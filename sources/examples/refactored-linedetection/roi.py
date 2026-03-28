"""
roi.py
------
ROISelector: handles mouse-driven region-of-interest cropping
on a named OpenCV window.  Completely decoupled from detection logic.
"""

import cv2
import numpy as np
from typing import Callable


class ROISelector:
    """
    Attach to an OpenCV window to let the user drag-select a crop region.

    Parameters
    ----------
    window_name : str
        The cv2 window this selector is bound to.
    source_image : np.ndarray
        The full-resolution BGR image to crop from.
    on_roi_change : Callable[[np.ndarray], None]
        Called whenever the ROI changes (new crop or reset to full image).
    min_drag_px : int
        Minimum drag size (both axes) to accept as a valid crop.
    """

    def __init__(
        self,
        window_name: str,
        source_image: np.ndarray,
        on_roi_change: Callable[[np.ndarray], None],
        min_drag_px: int = 100,
    ):
        self.window_name = window_name
        self.source = source_image
        self.on_roi_change = on_roi_change
        self.min_drag_px = min_drag_px

        self._cropping = False
        self._start: tuple[int, int] = (0, 0)

        cv2.setMouseCallback(window_name, self._callback)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start = (x, y)
            self._cropping = True

        elif event == cv2.EVENT_MOUSEMOVE and self._cropping:
            preview = self.source.copy()
            cv2.rectangle(preview, self._start, (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, preview)

        elif event == cv2.EVENT_LBUTTONUP:
            self._cropping = False
            x1, y1 = self._start
            x2, y2 = x, y

            if abs(x2 - x1) > self.min_drag_px and abs(y2 - y1) > self.min_drag_px:
                roi = self.source[
                    min(y1, y2) : max(y1, y2),
                    min(x1, x2) : max(x1, x2),
                ].copy()
            else:
                roi = self.source.copy()  # too small → reset to full

            self.on_roi_change(roi)
