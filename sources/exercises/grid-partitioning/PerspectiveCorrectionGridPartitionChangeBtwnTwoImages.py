import cv2
import numpy as np
from detectionAndCalcTools import *
import matplotlib.pyplot as plt

THRESHOLD = 15 # Step difference required for edge detection
STEP = 3 # Index hop width
GRAY_VALUE = 80 # Target gray value of the img
GRAY_THRESH = 0.13 # Percentage based +- hysteresis of the fixed gray value

HOUGH_THRESH = 50 # HoughlinesP threshold
MIN_LENGTH = 200 # HoughlinesP line min length
MAX_LINE_GAP = 30 # HoughlinesP max line gap

INTERSECTION_CLUSTER = 5 # Grid size for clustering multiple intersections nearby

# Parameters for grid partitioning
CELL_ROWS = 10
CELL_COLS = 10
CELL_THRESHOLD = 10
RECT_OFFSET = 5

# Read imgs
imgs = [cv2.imread("sources/exercises/grid-partitioning/image1_1.png"),
        cv2.imread("sources/exercises/grid-partitioning/image1_2_test.png")]

corners = None
detections = []

# Process both images
for img in imgs:
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = detectEdges(gray, THRESHOLD, STEP, GRAY_VALUE, GRAY_THRESH)
    lines, intersectPoints = findLinesAndIntersectPoints(edges, HOUGH_THRESH, MIN_LENGTH, MAX_LINE_GAP, INTERSECTION_CLUSTER)

    # If no corners yet found, find them from the first img in imgs
    if corners is None:
        corners = findCorners(intersectPoints)

    # Warp img based on detected corners
    warped = warp(img, corners)
    warpedGray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Detect edges again from warped img for more accurate intersection point detection
    warpedEdges = detectEdges(warpedGray, THRESHOLD, STEP, GRAY_VALUE, GRAY_THRESH)
    # Find intersection points from warped img (again - for more accurate detection)
    # Not using the fixed points given in the task assignment
    _, warpedIntersect = findLinesAndIntersectPoints(warpedEdges, HOUGH_THRESH, MIN_LENGTH, MAX_LINE_GAP, INTERSECTION_CLUSTER)

    # Store all required data for further use, testing, visualization etc
    detections.append({
        "img": img,
        "gray": gray,
        "edges": edges,
        "intersectPoints": intersectPoints,
        "warped": warped,
        "warpedEdges": warpedEdges,
        "warpedIntersect": warpedIntersect
        })

# Detect changes in cells/grids between the given imgs and highlight the changed cells
result = detectAndDrawChanges(detections[0]["warped"], detections[1]["warped"], (CELL_ROWS, CELL_COLS), CELL_THRESHOLD, RECT_OFFSET)

# Init matplotlib "visualization"
fig, axes = plt.subplots(2, 3, figsize=(10, 10))

# Iterate through the procecced imgs and then visualize the cells and changes
for c in range(len(detections)):
    # Warped
    axes[0, c].imshow(detections[c]["warped"][..., ::-1])
    axes[0, c].set_title(f"warped img{c+1}")

    # # Warped with intersection points
    # axes[c, 1].imshow(detections[c]["warped"][..., ::-1])
    # if detections[c]["warpedIntersect"] is not None:
    #     axes[c, 1].scatter(detections[c]["warpedIntersect"][:, 0], detections[c]["warpedIntersect"][:, 1], c='white', s=5, zorder=4)
    # axes[c, 1].set_title(f"grid intersections img{c+1}")

# Cells and changes in cells visualized on warped imgs
axes[1, 0].imshow(drawGrids(detections[0]["warped"], (CELL_ROWS, CELL_COLS), RECT_OFFSET)[..., ::-1])
axes[1, 0].set_title("cells img1")

axes[1, 1].imshow(drawGrids(detections[1]["warped"], (CELL_ROWS, CELL_COLS), RECT_OFFSET)[..., ::-1])
axes[1, 1].set_title("cells img2")

axes[1, 2].imshow(result[..., ::-1])
axes[1, 2].set_title("changes")
axes[0, 2].axis('off')

plt.tight_layout()
plt.show()























# fig = plt.figure(figsize=(20, 8))
# gs  = fig.add_gridspec(2, 3)

# ax00 = fig.add_subplot(gs[0, 0])
# ax10 = fig.add_subplot(gs[1, 0])
# ax01 = fig.add_subplot(gs[0, 1])
# ax11 = fig.add_subplot(gs[1, 1])
# ax_result = fig.add_subplot(gs[:, 2])

# # Warped images
# ax00.imshow(detections[0]["warped"][..., ::-1])
# ax00.set_title("warped img1")

# ax10.imshow(detections[1]["warped"][..., ::-1])
# ax10.set_title("warped img2")

# # Grid cells
# ax01.imshow(drawGrids(detections[0]["warped"], (CELL_ROWS, CELL_COLS), RECT_OFFSET)[..., ::-1])
# ax01.set_title("cells img1")

# ax11.imshow(drawGrids(detections[1]["warped"], (CELL_ROWS, CELL_COLS), RECT_OFFSET)[..., ::-1])
# ax11.set_title("cells img2")

# # Changes
# ax_result.imshow(result[..., ::-1])
# ax_result.set_title("changes")

# plt.tight_layout()
# plt.show()