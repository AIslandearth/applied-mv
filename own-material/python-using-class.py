import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools


class LineDetector:

    def __init__(self, path):
        self.image         = self._load(path)
        self.working_image = self.image.copy()
        self.roi_start     = None
        self.cropping      = False

        self.root       = tk.Tk()
        self.root.title("Line Detection")
        self.blur_scale = Scale(self.root, from_=1, to=31, orient=HORIZONTAL,
                                label="Blur size", command=self.update)
        self.blur_scale.set(5)
        self.blur_scale.pack(fill="x", padx=10, pady=6)

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        cv2.imshow("Image", self.image)

        self.update()
        self.root.after(30, self._cv_loop)
        self.root.mainloop()
        cv2.destroyAllWindows()

    # ── Load & resize to fit screen ──────────────
    def _load(self, path):
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()

        img = cv2.imread(path)
        img_h, img_w = img.shape[:2]

        if img_w > screen_w or img_h > screen_h:
            scale = min(screen_w / img_w, screen_h / img_h) * 0.9
            img   = cv2.resize(img, (int(img_w * scale), int(img_h * scale)))
        return img

    # ── Mouse callback for ROI ────────────────────
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.cropping  = True

        elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
            temp = self.image.copy()
            cv2.rectangle(temp, self.roi_start, (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            self.cropping  = False
            x1, y1 = self.roi_start
            x2, y2 = x, y

            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.working_image = self.image[min(y1,y2):max(y1,y2),
                                                min(x1,x2):max(x1,x2)].copy()
            else:
                self.working_image = self.image.copy()

            self.update()

    # ── Line math ─────────────────────────────────
    def _to_slope_intercept(self, x1, y1, x2, y2):
        if abs(x2 - x1) < 1e-6:
            return None, float(x1)
        k = (y2 - y1) / (x2 - x1)
        return k, y1 - k * x1

    def _intersect(self, k1, b1, k2, b2):
        if k1 is not None and k2 is not None:
            if abs(k1 - k2) < 1e-8:
                return None
            x = (b2 - b1) / (k1 - k2)
            return x, k1 * x + b1
        if k1 is None and k2 is not None:
            return b1, k2 * b1 + b2
        if k2 is None and k1 is not None:
            return b2, k1 * b2 + b1
        return None

    # ── Processing & display ──────────────────────
    def update(self, *args):
        h, w = self.working_image.shape[:2]

        blur = self.blur_scale.get()
        blur = blur if blur % 2 == 1 else blur + 1

        gray    = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
        edges   = cv2.Canny(blurred, 50, 150)

        cv2.imshow("Edges", edges)

        lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                  threshold=60, minLineLength=40, maxLineGap=15)
        display = self.working_image.copy()

        if lines is None:
            cv2.imshow("Result", display)
            cv2.waitKey(1)
            return

        slopes_intercepts = []
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            k, b = self._to_slope_intercept(x1, y1, x2, y2)
            slopes_intercepts.append((k, b))
            cv2.line(display, (x1, y1), (x2, y2), (0, 200, 0), 2)

        for (k1, b1), (k2, b2) in itertools.combinations(slopes_intercepts, 2):
            pt = self._intersect(k1, b1, k2, b2)
            if pt is None:
                continue
            ix, iy = pt
            if 0 <= ix <= w and 0 <= iy <= h:
                cv2.circle(display, (int(ix), int(iy)), 6, (0, 0, 255), -1)

        cv2.imshow("Result", display)
        cv2.waitKey(1)

    def _cv_loop(self):
        cv2.waitKey(1)
        self.root.after(30, self._cv_loop)


if __name__ == "__main__":
    LineDetector("image.jpg")   # ← change path
