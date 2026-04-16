import cv2
import numpy as np
import matplotlib.pyplot as plt
#from detectionAndCalcTools import *

THRESHOLD = 20 # Step difference required for edge detection
STEP = 3 # Index hop width
GRAY_VALUE = 80 # Target gray value of the img
GRAY_THRESH = 0.13 # Percentage based +- hysteresis of the fixed gray value

HOUGH_THRESH = 50 # HoughlinesP threshold
MIN_LENGTH = 30 # HoughlinesP line min length
MAX_LINE_GAP = 3 # HoughlinesP max line gap

# Read img and convert to grayscale
img = cv2.imread("sources/img/image1_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Detect edges using numpy and some gray value based masking after it
edges = detectEdges(gray, THRESHOLD, STEP, GRAY_VALUE, GRAY_THRESH)
# Find lines and intersections of the grayscale img
lines, intersectPoints = findLines(edges, HOUGH_THRESH, MIN_LENGTH, MAX_LINE_GAP)
# Find corners based on lines intersections found
corners, size = findCorners(intersectPoints)

# Warp the perspective using the detected grid corners as top left, top right, bottom right, bottom left
warped = warp(img, corners)
# Convert warped img to grayscale and detect edges again
warpedGray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warpedEdges = detectEdges(warpedGray, THRESHOLD, STEP, GRAY_VALUE, GRAY_THRESH)
# Find lines of the warped img based on detected edges
warpedLines, warpedIntersctPts = findLines(warpedEdges, HOUGH_THRESH, MIN_LENGTH, MAX_LINE_GAP)

# Init matplotlib "visualization"
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
# Grayscale
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("gray")
# Edges detected on grayscale
axes[1].imshow(edges, cmap='binary')
axes[1].set_title("edges")
# Lines found based on edge detection binary map
axes[2].imshow(img[...,::-1])
# Draw lines
drawLines(axes[2], lines, 'red', linewidth=1)

if corners is not None:
    cx = corners[:,0]
    cy = corners[:,1]
    #cx, cy = zip(*corners)
    axes[2].scatter(cx, cy, c='yellow', s=50 , zorder=5)
axes[2].set_title("corners")

axes[3].imshow(warped[...,::1])
axes[3].set_title("warped")

axes[4].imshow(warped[...,::-1])
drawLines(axes[4], warpedLines, 'red', linewidth=1)
axes[4].set_title("warped and lines")

#cv2.imshow("lines", draw_lines(img, lines))
plt.tight_layout()
plt.show()