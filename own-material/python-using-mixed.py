import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools


def resize_if_required(imagePath):
    root = tk.Tk()
    displayH = int(root.winfo_screenheight())
    displayW = int(root.winfo_screenwidth())
    root.destroy()

    img = cv2.imread(imagePath)  # use the parameter, not hardcoded path

    if img is None:
        print("Couldn't read the img")
        return None

    imgH, imgW = img.shape[:2]
    margin = 0.9

    if displayH < imgH or displayW < imgW:
        scaling = min(displayW / imgW, displayH / imgH)
        return cv2.resize(img, (int(imgW * scaling * margin), int(imgH * scaling * margin)))

    return img  # no resize needed


def to_slope_intercept(x1: int, y1: int, x2: int, y2: int) -> tuple:
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


def make_mouse_callback(state: dict, on_roi_selected: callable) -> callable:
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["roi_start"] = (x, y)
            state["cropping"]  = True

        elif event == cv2.EVENT_MOUSEMOVE and state["cropping"]:
            temp = state["image"].copy()
            cv2.rectangle(temp, state["roi_start"], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            state["cropping"] = False
            x1, y1 = state["roi_start"]
            x2, y2 = x, y

            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                state["imageROI"] = state["image"][min(y1,y2):max(y1,y2),
                                                   min(x1,x2):max(x1,x2)].copy()
            else:
                state["imageROI"] = state["image"].copy()

            on_roi_selected()

    return callback


def process_the_image(imgROI: np.ndarray, blur: int) -> np.ndarray:
    blur    = blur if blur % 2 == 1 else blur + 1
    h, w    = imgROI.shape[:2]
    gray    = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges   = cv2.Canny(blurred, 100, 200)

    cv2.imshow("Edges", edges)

    lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                              threshold=60, minLineLength=40, maxLineGap=15)
    display = imgROI.copy()

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


def main():
    image = resize_if_required("sources/img/testImage3.png")
    if image is None:
        return

    state = {"image": image, "imageROI": image.copy(), "roi_start": None, "cropping": False}

    root = tk.Tk()
    root.title("Line Detection")

    blur_scale = Scale(root, from_=1, to=31, orient=HORIZONTAL, label="Blur size")
    blur_scale.set(7)
    blur_scale.pack(fill="x", padx=10, pady=6)  # was commented out — slider invisible without this

    def update_image(*args):
        result = process_the_image(state["imageROI"], blur_scale.get())  # use state, not local var
        cv2.imshow("Result", result)
        cv2.waitKey(1)

    blur_scale.config(command=update_image)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", make_mouse_callback(state, on_roi_selected=update_image))
    cv2.imshow("Image", image)

    update_image()

    def cv_loop():
        cv2.waitKey(1)
        root.after(30, cv_loop)

    root.after(30, cv_loop)  # keep OpenCV windows responsive
    root.mainloop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
