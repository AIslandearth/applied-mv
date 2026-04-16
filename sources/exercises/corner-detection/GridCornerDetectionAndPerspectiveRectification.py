import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectionAndCalcTools import *

THRESHOLD = 15 # Step difference required for edge detection
STEP = 3 # Index hop width
GRAY_VALUE = 80 # Target gray value of the img
GRAY_THRESH = 0.13 # Percentage based +- hysteresis of the fixed gray value

HOUGH_THRESH = 50 # HoughlinesP threshold
MIN_LENGTH = 200 # HoughlinesP line min length
MAX_LINE_GAP = 30 # HoughlinesP max line gap

CLUSTER_GRID = 5 # Grid size for clustering multiple intersections nearby

# Read img and convert to grayscale
img = cv2.imread("sources/img/image1_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Detect edges using numpy and some gray value based masking after it
edges = detectEdges(gray, THRESHOLD, STEP, GRAY_VALUE, GRAY_THRESH)
# Find lines and intersections of the grayscale img
lines, intersectPoints = findLines(edges, HOUGH_THRESH, MIN_LENGTH, MAX_LINE_GAP, CLUSTER_GRID)
# Find corners based on lines intersections found
corners = findCorners(intersectPoints)

# Warp the perspective using the detected grid corners as top left, top right, bottom right, bottom left
warped = warp(img, corners)
# Convert warped img to grayscale and detect edges again
warpedGray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warpedEdges = detectEdges(warpedGray, THRESHOLD, STEP, GRAY_VALUE, GRAY_THRESH)
# Find lines of the warped img based on detected edges
warpedLines, warpedIntersctPts = findLines(warpedEdges, HOUGH_THRESH, MIN_LENGTH, MAX_LINE_GAP, CLUSTER_GRID)

# Init matplotlib "visualization"
fig, axes = plt.subplots(2, 3, figsize=(20, 5))
# Grayscale
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("gray")

# Edges detected on grayscale
axes[0, 1].imshow(edges, cmap='binary')
axes[0, 1].set_title("edges")

# Show all detected intersections
axes[0, 2].imshow(img[...,::-1])
if intersectPoints is not None:
    axes[0, 2].scatter(intersectPoints[:,0], intersectPoints[:,1], c='white', s=5, zorder=4)
axes[0, 2].set_title("intersections")

# Draw the found lines based on edge detection binary map
axes[1, 0].imshow(img[...,::-1])
drawLines(axes[1, 0], lines, 'red', linewidth=1)

# Mark detected grid corners
if corners is not None:
    cx = corners[:,0]
    cy = corners[:,1]
    #cx, cy = zip(*corners)
    axes[1, 0].scatter(cx, cy, c='yellow', s=50 , zorder=5)
axes[1, 0].set_title("corners")

# Warped img
axes[1, 1].imshow(warped[...,::1])
axes[1, 1].set_title("warped")

# Warped with lines
axes[1, 2].imshow(warped[...,::-1])
drawLines(axes[1, 2], warpedLines, 'red', linewidth=1)
axes[1, 2].set_title("warped and lines")

#cv2.imshow("lines", draw_lines(img, lines))
plt.tight_layout()
plt.show()