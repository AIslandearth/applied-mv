import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools

# Size of the display in use for img initial resizing
root = tk.Tk()
displayH = root.winfo_screenheight()
displayW = root.winfo_screenwidth()
# No need after display variables assigned
root.destroy()

# Read img from file
image = cv2.imread("sources/img/testImage3.png")
# img height and width, order = (height, width, channels)
imgH, imgW = image.shape[:2]

# Resize if image is larger than the display
if imgW > displayW or imgH > displayH:
    # 0.9 = "safe" margin for the resized window
    # Calculate scaling factor for resizing using width and height + margin
    scaling = min(displayW / imgW, displayH / imgH) * 0.9
    # Resized img
    image = cv2.resize(image, (int(imgW * scaling), int(imgH * scaling)))

# Copy the "default" image as the ROI
ROI = image.copy()
ROI_start = None
# Full window ROI at start
cropping = False

# Calc slope "k" of line in process
# y = kx + b
def to_slope_intercept(x1, y1, x2, y2):
    if abs(x2 - x1) < 1e-8:
        return None, float(x1)
    k = (y2 - y1) / (x2 - x1)
    # Determine the "y" axis intercept // slope formula
    return k, y1 - k * x1


def line_intersections(slopes_intercepts):
    intersections = []
    # Calc intersections of two lines
    for (i, (k1, b1)), (j, (k2, b2)) in itertools.combinations(enumerate(slopes_intercepts), 2):
        # If two lines exists/detected
        if k1 is not None and k2 is not None:
            # Parallel lines
            if abs(k1 - k2) < 1e-8:
                continue
            # Intersection
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
        # k1 line is vertical, line equations
        elif k1 is None and k2 is not None:
            x = b1
            y = k2 * x + b2
        # k2 line is vertical, samesame
        elif k2 is None and k1 is not None:
            x = b2
            y = k1 * x + b1
        # keep going
        else:
            continue
        # Add to list
        intersections.append((x, y, i + 1, j + 1))
    return intersections


def mouse_callback(event, x, y, flags, param):
    global ROI, roi_start, cropping

    # Press left btn on mouse and detect start point and new ROI "crop" set true
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        cropping = True

    # When moving the mouse and cropping = true
    # Draw rectangle shape to define new ROI
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp = image.copy()
        cv2.rectangle(temp, roi_start, (x, y), (0, 255, 0), 2)
        cv2.imshow("ROI image", temp)
    # Release mouse left btn New ROI's start and end coordinates
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        x1, y1 = roi_start
        x2, y2 = x, y
        # if mouse movement more than
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            ROI = image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)].copy()
        else:
            ROI = image.copy()
        # Update the gray, blur, edge and ROI imshow
        update_image()


def update_image(*args):
    # Tkinter read slider pos
    blur = blur_scale.get()
    # Cannot be even, has to be odd
    blur = blur if blur % 2 == 1 else blur + 1

    # Gray, blur, edges
    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    # High contrast img -> seems to be ok with 200, 300
    edges = cv2.Canny(blurred, 200, 300)

    # Show grayscale, blurred and edges img
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edges)
    # Copy ROI to display in window
    display = ROI.copy()
    
    # Use probabilistic hough lines for line detection
    # img = edges, resolution/precision = 1px, angular res/prec in rad = 1deg, threshold = min count of edge points to detect line,
    # min line length in pixels, max gap in pixels btwn two points to count/merge as line
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=20)
    
#    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
#
#   if lines is not None:
#       for rho, theta in lines[:, 0]:
#           # Convert polar to two cartesian points
#           a  = np.cos(theta)
#           b  = np.sin(theta)
#           x0 = a * rho
#           y0 = b * rho
#           # Extend far in both directions to draw a full line
#           x1 = int(x0 + 1000 * (-b))
#           y1 = int(y0 + 1000 * (a))
#           x2 = int(x0 - 1000 * (-b))
#           y2 = int(y0 - 1000 * (a))
#           cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # If no lines detected, show img and and wait 1ms for "rendering", TKinter handles the loop as "main"
    if lines is None:
        cv2.imshow("", display)
        cv2.waitKey(1)
        return

    def line_length(seg):
        # Calc length
        x1, y1, x2, y2 = seg[0]
        # using pythagora
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    # Sort longest three [:3] lines, starting from the most longest one
    lines = sorted(lines, key=line_length, reverse=True)[:3]

    # Def list
    slopes_intercepts = []
    # Calc slope k 
    for seg in lines:
        # unpack "start" and "end" point from line segment
        x1, y1, x2, y2 = seg[0]
        # Calc slope, y = kx + b
        k, b = to_slope_intercept(x1, y1, x2, y2)
        # Add equation to list
        slopes_intercepts.append((k, b))
        # Display pts w/ red, thickness 2
        cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Get ROI height and width
    h, w = ROI.shape[:2]
    # Compute intersecting line pairs
    intersections = line_intersections(slopes_intercepts)
    # Print on console "Lines this and that intersects at x, y"
    for x, y, i, j in intersections:
        print(f"Lines {i} and {j} intersect at ({x:.2f}, {y:.2f})")
        # Check that the intersection point is in image boundaries
        # for displaying the black dot as the intersection point
        if 0 <= x <= w and 0 <= y <= h:
            cv2.circle(display, (int(x), int(y)), 6, (0, 0, 0), -1)

    # Show img with the detected lines and intersections
    # and the "orig" img
    cv2.imshow("ROI with detected lines", display)
    cv2.imshow("Default image", image)
    cv2.waitKey(1)


root = tk.Tk()
root.title("Line detection")

blur_scale = Scale(root, from_=1, to=10, orient=HORIZONTAL, label="Blur size", command=update_image)
blur_scale.set(1)
blur_scale.pack()

cv2.namedWindow("Default image")
cv2.setMouseCallback("Default image", mouse_callback)

root.update()
update_image()

root.mainloop()
cv2.destroyAllWindows()
