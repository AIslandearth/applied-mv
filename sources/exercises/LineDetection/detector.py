"""
detector.py
-----------
LineDetector: wraps OpenCV edge detection + Hough line detection.
Owns no UI state — takes an image, returns annotated results.
"""

import cv2
import numpy as np

from .geometry import to_slope_intercept, line_intersections


class LineDetector:
    """
    Detects the N longest lines in an image using Canny + HoughLinesP,
    computes pairwise intersections, and returns an annotated copy.

    Parameters
    ----------
    blur_size : int
        Gaussian kernel size (will be forced odd, min 1).
    canny_low : int
        Lower threshold for Canny edge detection.
    canny_high : int
        Upper threshold for Canny edge detection.
    hough_threshold : int
        Minimum Hough accumulator votes to accept a line.
    min_line_length : int
        Minimum segment length in pixels.
    max_line_gap : int
        Maximum allowed gap (px) to merge collinear segments.
    top_n : int
        Keep only the N longest detected segments.
    """

    def __init__(
        self,
        blur_size: int = 7,
        canny_low: int = 200,
        canny_high: int = 300,
        hough_threshold: int = 80,
        min_line_length: int = 60,
        max_line_gap: int = 20,
        top_n: int = 3,
    ):
        self.blur_size = blur_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.top_n = top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> dict:
        """
        Run the full pipeline on *image* (BGR).

        Returns
        -------
        dict with keys:
            gray        – grayscale image
            blurred     – blurred grayscale image
            edges       – Canny edge map
            annotated   – BGR copy of *image* with lines + intersections drawn
            lines       – list of raw segments [[x1,y1,x2,y2], …]
            intersections – list of (x, y, i, j) tuples
        """
        blur = self._ensure_odd(self.blur_size)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        annotated = image.copy()
        segments: list = []
        intersections: list = []

        if raw is not None:
            sorted_lines = sorted(raw, key=self._segment_length, reverse=True)[: self.top_n]
            slopes_intercepts = []

            for seg in sorted_lines:
                x1, y1, x2, y2 = seg[0]
                segments.append([x1, y1, x2, y2])
                k, b = to_slope_intercept(x1, y1, x2, y2)
                slopes_intercepts.append((k, b))
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

            h, w = image.shape[:2]
            intersections = line_intersections(slopes_intercepts)
            for x, y, i, j in intersections:
                print(f"Lines {i} and {j} intersect at ({x:.2f}, {y:.2f})")
                if 0 <= x <= w and 0 <= y <= h:
                    cv2.circle(annotated, (int(x), int(y)), 6, (0, 0, 0), -1)

        return {
            "gray": gray,
            "blurred": blurred,
            "edges": edges,
            "annotated": annotated,
            "lines": segments,
            "intersections": intersections,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_length(seg) -> float:
        x1, y1, x2, y2 = seg[0]
        return np.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _ensure_odd(n: int) -> int:
        n = max(1, int(n))
        return n if n % 2 == 1 else n + 1
