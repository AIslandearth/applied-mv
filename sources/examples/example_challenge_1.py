import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from PIL import Image, ImageTk
import itertools


roi_pts = [(5, 349), (1014, 581)]
cropping = False

x_multiplier = 0.25
y_multiplier = 0.25    
image = cv2.imread("sources/img/example_challenge_1.jpg")
resized_image = cv2.resize(image, (int(image.shape[1] * x_multiplier), int(image.shape[0] * y_multiplier)))

x1, y1 = roi_pts[0]
x2, y2 = roi_pts[1]
roi = resized_image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

def line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2

    return a, b, c


def ransac_line(points, n_iter=1000, threshold=1.0, min_inliers=20):
    best_inliers = []
    best_model = None

    n_points = len(points)
    if(n_points < 2):
        return None, points 
    
    for _ in range(n_iter):
        i1, i2 = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[i1], points[i2]

        a, b, c = line_from_points(p1, p2)

        distances = np.abs(a*points[:,0] + b*points[:,1] + c) / np.sqrt(a*a + b*b)

        inliers = points[distances < threshold]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (a, b, c)

    if len(best_inliers) < min_inliers:
        return None, points

    outliers = np.array([p for p in points if not any((p == best_inliers).all(1))])

    return best_model, best_inliers, outliers

def find_multiple_lines(points, n_lines=4, threshold=1.0):
    remaining = points.copy()
    lines = []

    for i in range(n_lines):
        result = ransac_line(remaining, threshold=threshold)

        if result[0] is None:
            break

        model, inliers, remaining = result
        lines.append((model, inliers))

        print(f"Viiva {i+1}: {len(inliers)} pistettä")

    return lines, remaining


def findlines(image):
    height, width = image.shape[:2]
    points =[]
    for y in range(0, height, 5):
        previous_color = None
        for x in range(width):
            color = image[y, x]
            if previous_color is not None:
                difference = abs(int(color) - int(previous_color))
                if(difference > 30):
                    points.append((x, y))   
            previous_color = color
    return np.array(points)

def line_to_slope_intercept(a, b, c):
    if abs(b) < 1e-8:
        return None  

    k = -a / b
    intercept = -c / b
    return k, intercept

def line_intersections(lines):
    intersections = []

    for (i, (k1, b1)), (j, (k2, b2)) in itertools.combinations(enumerate(lines), 2):
        if k1 is not None and k2 is not None:
            if abs(k1 - k2) < 1e-8:  
                continue
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1

        elif k1 is None:
            x = b1
            y = k2 * x + b2

        elif k2 is None:
            x = b2
            y = k1 * x + b1
        intersections.append((x, y, i+1, j+1))

    return intersections


def mouse_callback(event, x, y, flags, param):
    global roi_pts, cropping, resized_image, roi

    # Hiiren vasen nappi painettu -> aloita rajaaminen
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [(x, y)]       
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp_image = resized_image.copy()
        cv2.rectangle(temp_image, roi_pts[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("resized_image", temp_image) 
        
    elif event == cv2.EVENT_LBUTTONUP:
        roi_pts.append((x, y))
        cropping = False
        print(roi_pts)

        # Rajataan ROI
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]

        roi = resized_image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        update_image()

        cv2.imshow("ROI", roi)

def update_image(*args):
    blur_size = blur_size_scale.get()
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  
    
    #resized_image = cv2.resize(image, (int(image.shape[1] * x_multiplier), int(image.shape[0] * y_multiplier)))
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)    
    blurred_image = cv2.GaussianBlur(gray_image, (blur_size, blur_size), 0)  
    edges = cv2.Canny(blurred_image, 100, 200)
    cv2.imshow("blurred_image", blurred_image)
    cv2.imshow("edges", edges)
    #cv2.imwrite("edges.jpg", edges)
    display_image = roi.copy()

    points = findlines(edges)
    lines, leftover = find_multiple_lines(points, n_lines=6, threshold=7.5)
    slopes_intercepts = []
    for i, (model, _) in enumerate(lines):
        a, b, c = model
        result = line_to_slope_intercept(a, b, c)

        if result is not None:
            k, b0 = result
            slopes_intercepts.append((k, b0))

    black = np.zeros_like(image)
    for x in range(image.shape[1]):
        for k, b0 in slopes_intercepts:
            y = int(k * x + b0)
            if 0 <= y < image.shape[0]:
                cv2.circle(display_image, (x, y), 1, (0, 0, 255), -1)   
                #black[y, x] = 255           

    intersections = line_intersections(slopes_intercepts)
    print("Intersections:")
    for x, y, i, j in intersections:
        print(f"Lines {i} ja {j} intersect at point ({x:.2f}, {y:.2f})")

    cv2.imshow("display_image", display_image)
    cv2.imshow("resized_image", resized_image)
    cv2.waitKey(1)  


# --------------------- TKINTER GUI ---------------------
root = tk.Tk()
root.title("HoughLinesP Controls (Tkinter sliders + cv2.imshow)")

blur_size_scale = Scale(root, from_=1, to=100, orient=HORIZONTAL, label="blur size", command=update_image)
blur_size_scale.set(7)
blur_size_scale.pack()


cv2.namedWindow("resized_image")
cv2.setMouseCallback("resized_image", mouse_callback)

update_image()

# Tkinterin looppi
root.mainloop()

cv2.destroyAllWindows()