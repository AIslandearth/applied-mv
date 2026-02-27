import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools


# --------------------- INIT ---------------------
root = tk.Tk()
displayH = root.winfo_screenheight()
displayW = root.winfo_screenwidth()
root.destroy()

image = cv2.imread("sources/img/example_challenge_1.jpg")
imgH, imgW = image.shape[:2]

if imgW > displayW or imgH > displayH:
    scaling = min(displayW / imgW, displayH / imgH) * 0.9
    image = cv2.resize(image, (int(imgW * scaling), int(imgH * scaling)))

roi      = image.copy()
roi_start = None
cropping  = False


# --------------------- LINE MATH ---------------------
def to_slope_intercept(x1, y1, x2, y2):
    if abs(x2 - x1) < 1e-8:
        return None, float(x1)          # vertical line: x = const
    k = (y2 - y1) / (x2 - x1)
    return k, y1 - k * x1


def line_length(seg):
    x1, y1, x2, y2 = seg[0]
    return np.hypot(x2 - x1, y2 - y1)  # cleaner than manual sqrt


def line_intersections(slopes_intercepts):
    intersections = []
    for (i, (k1, b1)), (j, (k2, b2)) in itertools.combinations(enumerate(slopes_intercepts), 2):
        if k1 is not None and k2 is not None:
            if abs(k1 - k2) < 1e-8:
                continue                # parallel
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
        elif k1 is None and k2 is not None:
            x, y = b1, k2 * b1 + b2    # line1 vertical
        elif k2 is None and k1 is not None:
            x, y = b2, k1 * b2 + b1    # line2 vertical
        else:
            continue                    # both vertical
        intersections.append((x, y, i + 1, j + 1))
    return intersections


# --------------------- MOUSE ---------------------
def mouse_callback(event, x, y, flags, param):
    global roi, roi_start, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        cropping  = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp = image.copy()
        cv2.rectangle(temp, roi_start, (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        cropping  = False
        x1, y1    = roi_start
        x2, y2    = x, y
        roi       = image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)].copy() \
                    if abs(x2-x1) > 10 and abs(y2-y1) > 10 else image.copy()
        update_image()


# --------------------- PROCESSING ---------------------
def update_image(*args):
    blur    = blur_scale.get()
    blur    = blur if blur % 2 == 1 else blur + 1
    h, w    = roi.shape[:2]

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges   = cv2.Canny(blurred, 150, 300)

    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges",   edges)

    lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                              threshold=80, minLineLength=60, maxLineGap=10)
    display = roi.copy()

    if lines is None:
        cv2.imshow("Result", display)
        cv2.waitKey(1)
        return

    top2 = sorted(lines, key=line_length, reverse=True)[:2]

    slopes_intercepts = []
    for seg in top2:
        x1, y1, x2, y2 = seg[0]
        k, b = to_slope_intercept(x1, y1, x2, y2)
        slopes_intercepts.append((k, b))
        cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for x, y, i, j in line_intersections(slopes_intercepts):
        print(f"Lines {i} and {j} intersect at ({x:.2f}, {y:.2f})")
        if 0 <= x <= w and 0 <= y <= h:
            cv2.circle(display, (int(x), int(y)), 8, (0, 0, 0), -1)

    cv2.imshow("Result", display)
    cv2.imshow("Image",  image)
    cv2.waitKey(1)


# --------------------- GUI ---------------------
root = tk.Tk()
root.title("Line Detection")

blur_scale = Scale(root, from_=1, to=100, orient=HORIZONTAL, label="Blur size", command=update_image)
blur_scale.set(5)
blur_scale.pack()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

root.update()
update_image()

root.mainloop()
cv2.destroyAllWindows()
