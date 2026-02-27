import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools

# ── Globals ──────────────────────────────────
image         = None
working_image = None
roi_start     = None
cropping      = False

# ── Load & resize to fit screen ──────────────
def load_image(path):
    global image, working_image

    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    img = cv2.imread(path)
    img_h, img_w = img.shape[:2]

    if img_w > screen_w or img_h > screen_h:
        scale = min(screen_w / img_w, screen_h / img_h) * 0.9
        img   = cv2.resize(img, (int(img_w * scale), int(img_h * scale)))

    image         = img
    working_image = img.copy()

# ── Mouse callback for ROI ────────────────────
def mouse_callback(event, x, y, flags, param):
    global roi_start, cropping, working_image

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        cropping  = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp = image.copy()
        cv2.rectangle(temp, roi_start, (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        x1, y1   = roi_start
        x2, y2   = x, y

        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            working_image = image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)].copy()
        else:
            working_image = image.copy()

        update()

# ── Line math ─────────────────────────────────
def to_slope_intercept(x1, y1, x2, y2):
    if abs(x2 - x1) < 1e-6:
        return None, float(x1)
    k = (y2 - y1) / (x2 - x1)
    return k, y1 - k * x1

def intersect(k1, b1, k2, b2):
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

# ── Main processing & display ─────────────────
def update(*args):
    h, w = working_image.shape[:2]

    blur = blur_scale.get()
    blur = blur if blur % 2 == 1 else blur + 1

    gray    = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    cv2.imshow("Edges", edges)

    lines   = cv2.HoughLinesP(edges, 1, np.pi/180,
                              threshold=60, minLineLength=40, maxLineGap=15)
    display = working_image.copy()

    if lines is None:
        cv2.imshow("Result", display)
        cv2.waitKey(1)
        return

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

    cv2.imshow("Result", display)
    cv2.waitKey(1)

# ── Tkinter GUI ───────────────────────────────
root = tk.Tk()
root.title("Line Detection")

blur_scale = Scale(root, from_=1, to=31, orient=HORIZONTAL,
                   label="Blur size", command=update)
blur_scale.set(5)
blur_scale.pack(fill="x", padx=10, pady=6)

# ── Start ─────────────────────────────────────
load_image("image.jpg")   # ← change path

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)
cv2.imshow("Image", image)

update()

def cv_loop():
    cv2.waitKey(1)
    root.after(30, cv_loop)

root.after(30, cv_loop)
root.mainloop()
cv2.destroyAllWindows()
