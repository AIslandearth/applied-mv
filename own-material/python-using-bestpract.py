import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools
from dataclasses import dataclass
from typing import Optional


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    image:         np.ndarray
    working_image: np.ndarray
    roi_start:     Optional[tuple] = None
    cropping:      bool            = False


# ── Image loading ─────────────────────────────────────────────────────────────

def get_screen_size() -> tuple[int, int]:
    root = tk.Tk()
    size = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return size


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")

    screen_w, screen_h = get_screen_size()
    img_h, img_w       = img.shape[:2]

    if img_w > screen_w or img_h > screen_h:
        scale = min(screen_w / img_w, screen_h / img_h) * 0.9
        img   = cv2.resize(img, (int(img_w * scale), int(img_h * scale)))

    return img


# ── Line math ─────────────────────────────────────────────────────────────────

def to_slope_intercept(x1: int, y1: int, x2: int, y2: int) -> tuple:
    if abs(x2 - x1) < 1e-6:
        return None, float(x1)      # vertical: x = const
    k = (y2 - y1) / (x2 - x1)
    return k, y1 - k * x1


def intersect(k1, b1, k2, b2) -> Optional[tuple]:
    if k1 is not None and k2 is not None:
        if abs(k1 - k2) < 1e-8:
            return None             # parallel
        x = (b2 - b1) / (k1 - k2)
        return x, k1 * x + b1
    if k1 is None and k2 is not None:
        return b1, k2 * b1 + b2    # line1 vertical
    if k2 is None and k1 is not None:
        return b2, k1 * b2 + b1    # line2 vertical
    return None


# ── Processing ────────────────────────────────────────────────────────────────

def process(working_image: np.ndarray, blur: int) -> np.ndarray:
    blur    = blur if blur % 2 == 1 else blur + 1
    h, w    = working_image.shape[:2]
    gray    = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    cv2.imshow("Edges", edges)

    lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                              threshold=60, minLineLength=40, maxLineGap=15)
    display = working_image.copy()

    if lines is None:
        return display

    slopes_intercepts = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        k, b = to_slope_intercept(x1, y1, x2, y2)
        slopes_intercepts.append((k, b))
        cv2.line(display, (x1, y1), (x2, y2), (0, 200, 0), 2)

    for (k1, b1), (k2, b2) in itertools.combinations(slopes_intercepts, 2):
        pt = intersect(k1, b1, k2, b2)
        if pt is None:
            continue
        ix, iy = pt
        if 0 <= ix <= w and 0 <= iy <= h:
            cv2.circle(display, (int(ix), int(iy)), 6, (0, 0, 255), -1)

    return display


# ── Mouse callback (closure injects state) ────────────────────────────────────

def make_mouse_callback(state: AppState, on_roi_selected: callable) -> callable:
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.roi_start = (x, y)
            state.cropping  = True

        elif event == cv2.EVENT_MOUSEMOVE and state.cropping:
            temp = state.image.copy()
            cv2.rectangle(temp, state.roi_start, (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            state.cropping = False
            x1, y1 = state.roi_start
            x2, y2 = x, y

            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                state.working_image = state.image[min(y1,y2):max(y1,y2),
                                                  min(x1,x2):max(x1,x2)].copy()
            else:
                state.working_image = state.image.copy()

            on_roi_selected()

    return callback


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    image = load_image("image.jpg")   # ← change path
    state = AppState(image=image, working_image=image.copy())

    root       = tk.Tk()
    root.title("Line Detection")
    blur_scale = Scale(root, from_=1, to=31, orient=HORIZONTAL, label="Blur size")
    blur_scale.set(7)
    #blur_scale.pack(fill="x", padx=10, pady=6)

    def update_image(*args):
        result = process(state.working_image, blur_scale.get())
        cv2.imshow("Result", result)
        cv2.waitKey(1)

    blur_scale.config(command=update_image)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", make_mouse_callback(state, on_roi_selected=update_image))
    cv2.imshow("Image", image)

    update_image()

    #def cv_loop():
    #    cv2.waitKey(1)
    #    root.after(30, cv_loop)

    #root.after(30, cv_loop)
    root.mainloop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
