"""
app.py
------
App: orchestrates Tkinter controls + OpenCV windows.
Wires together ROISelector and LineDetector.
"""

import tkinter as tk
from tkinter import Scale, HORIZONTAL

import cv2
import numpy as np

from .detector import LineDetector
from .roi import ROISelector


class App:
    """
    Main application class.

    Parameters
    ----------
    image_path : str
        Path to the source image file.
    detector : LineDetector | None
        A pre-configured LineDetector instance.  A default one is created
        if None is passed.
    display_margin : float
        Fraction of screen to keep free when auto-resizing a large image.
    """

    WIN_SOURCE = "Default image"
    WIN_ROI    = "ROI with detected lines"
    WIN_GRAY   = "Grayscale"
    WIN_BLUR   = "Blurred"
    WIN_EDGES  = "Edges"

    def __init__(
        self,
        image_path: str,
        detector: LineDetector | None = None,
        display_margin: float = 0.9,
    ):
        self.detector = detector or LineDetector()
        self._roi: np.ndarray | None = None

        # --- load & optionally downscale image ---
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        self.image = self._fit_to_screen(image, display_margin)

        # --- Tkinter root & blur slider ---
        self.root = tk.Tk()
        self.root.title("Line detection")

        self.blur_scale = Scale(
            self.root,
            from_=1,
            to=10,
            orient=HORIZONTAL,
            label="Blur size",
            command=self._on_slider_change,
        )
        self.blur_scale.set(self.detector.blur_size)
        self.blur_scale.pack()

        # --- OpenCV windows ---
        cv2.namedWindow(self.WIN_SOURCE)

        # ROI starts as the full image
        self._roi = self.image.copy()
        self._roi_selector = ROISelector(
            window_name=self.WIN_SOURCE,
            source_image=self.image,
            on_roi_change=self._on_roi_change,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the event loop."""
        self.root.update()
        self._refresh()
        self.root.mainloop()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_slider_change(self, *_) -> None:
        self.detector.blur_size = self.blur_scale.get()
        self._refresh()

    def _on_roi_change(self, roi: np.ndarray) -> None:
        self._roi = roi
        self._refresh()

    # ------------------------------------------------------------------
    # Core refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        result = self.detector.detect(self._roi)

        cv2.imshow(self.WIN_GRAY,   result["gray"])
        cv2.imshow(self.WIN_BLUR,   result["blurred"])
        cv2.imshow(self.WIN_EDGES,  result["edges"])
        cv2.imshow(self.WIN_ROI,    result["annotated"])
        cv2.imshow(self.WIN_SOURCE, self.image)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_to_screen(image: np.ndarray, margin: float) -> np.ndarray:
        tmp = tk.Tk()
        screen_w = tmp.winfo_screenwidth()
        screen_h = tmp.winfo_screenheight()
        tmp.destroy()

        h, w = image.shape[:2]
        if w > screen_w or h > screen_h:
            scale = min(screen_w / w, screen_h / h) * margin
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image
