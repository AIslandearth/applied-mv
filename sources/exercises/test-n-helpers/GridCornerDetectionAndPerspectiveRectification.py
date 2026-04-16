import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from findEdgesAndCorners import *

THRESHOLD = 13
STEP = 3
MIN_WIDTH = 3
MAX_WIDTH = 10
GRAY_VALUE = 100
MIN_LENGTH = 80
MAX_LENGTH = 100


img = cv2.imread("sources/img/image1_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

H, W = gray.shape

edges = detect_edges(gray, THRESHOLD, STEP, MIN_WIDTH, MAX_WIDTH, GRAY_VALUE)
corners, size = find_corners(fit_lines(edges, MIN_LENGTH, MAX_LENGTH), H, W)
warped = warp(img, corners, size) if corners is not None else img
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

warped_edges = detect_edges(warped_gray, THRESHOLD, STEP, MIN_WIDTH, MAX_WIDTH, GRAY_VALUE)
lines = fit_lines(warped_edges, MIN_LENGTH, MAX_LENGTH)

mid_y = warped_gray.shape[0] // 2
fig, axes = plt.subplots(2, 3)

axes[0, 0].plot(gray[H // 2])
axes[0, 0].set_title("gray")

axes[0, 1].plot(edges[H // 2])
axes[0, 1].set_ylim(-0.5, 1.5); axes[0, 1].set_title("edge diff profile")

axes[0, 2].imshow(edges, cmap="binary")
axes[0, 2].axhline(y=H // 2, color="r", linestyle="--")
axes[0, 2].set_title("edges");

axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
if corners:
    axes[1, 0].plot(*zip(*corners), "oy", ms=8)
    for label, (x, y) in zip(["TL","TR","BR","BL"], corners):
        axes[1, 0].text(x+6, y+6, label, color="yellow", fontsize=10)
axes[1, 0].set_title("corners");

axes[1, 1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
for p0, p1, _ in lines:
    axes[1, 1].plot([p0[0],p1[0]], [p0[1],p1[1]], "w-", lw=2)
axes[1, 1].set_title(f"warped + lines ({len(lines)})");

axes[1, 2].imshow(warped_edges, cmap="binary")
axes[1, 2].axhline(y=mid_y, color="r", linestyle="--")
axes[1, 2].set_title("warped edge mask");

plt.tight_layout()
plt.show()


# fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 8))

# ax0.plot(mid_pass_gray)
# ax0.set_title("gray profile")

# ax1.plot(mid_pass_diff)
# ax1.set_ylim(-0.5, 1.5)
# ax1.set_title("edge profile")

# ax2.imshow(line_centers, cmap="binary")
# ax2.axhline(y=gray.shape[0] // 2, color="r", linestyle="--")
# ax2.set_title("line centers / edge mask")
# ax2.axis("off")

# ax3.imshow(gray, cmap="gray")
# if grid_corners is not None:
#     ax3.plot(grid_corners[:, 0], grid_corners[:, 1], "oy")
# ax3.set_title("corners")
# ax3.axis("off")

# plt.tight_layout()
# plt.show()







#cv2.imshow("original", img)
# cv2.imshow("edges", draw(gray, centers))
# cv2.imshow("corners", draw(gray, centers, corners))
# cv2.imshow("warped", warp(gray, corners))

cv2.waitKey(0)
cv2.destroyAllWindows()


# from calcCorners_2 import *
# from findEdges_2 import *


# # StepDiff
# SD_THRESH = 30
# SD_STEP   = 3
# SD_SLOT = 10
# SD_SLOT_THRESHOLD = 30

# # Derivative
# DV_THRESH = 60
# DV_STEP   = 5
# DV_BLUR   = 9
# DV_SLOT = 1
# DV_SLOT_THRESHOLD = 60

# #sd = StepDiff(gray, SD_THRESH, SD_STEP, SD_SLOT, SD_SLOT_THRESHOLD)
# #dc = DenseCorners(sd)
# #dv = Derivative(gray, DV_THRESH, DV_STEP, DV_BLUR, DV_SLOT, DV_SLOT_THRESHOLD)
# #dc = SimpleCorners(dv)
# dvv = Derivative(gray, DV_THRESH, DV_STEP, DV_BLUR, DV_SLOT, DV_SLOT_THRESHOLD)
# dc = DenseCorners(dvv)
# # Original + edges
# edges = dvv.draw()
# dc.draw_corners(edges)

# # Warped
# warped = warpToCorners(gray, dc.corners)
# warped_edges = None

# if warped is not None:
#     #sd_warped = StepDiff(warped, SD_THRESH, SD_STEP, SD_SLOT, SD_SLOT_THRESHOLD)
#     #dc_warped = DenseCorners(sd_warped)
#     #warped_edges = sd_warped.draw()
#     #dc_warped.draw_corners(warped_edges)
    
#     dv_warped = Derivative(warped, DV_THRESH, DV_STEP, DV_BLUR, DV_SLOT, DV_SLOT_THRESHOLD)
#     sc_warped = DenseCorners(dv_warped)
#     warped_edges = dv_warped.draw()
#     sc_warped.draw_corners(warped_edges)

# # Display
# panels = {
#     "original":          cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
#     "original + edges":  edges,
#     "warped":            cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR) if warped is not None else None,
#     "warped + edges":    warped_edges,
# }

# for title, panel in panels.items():
#     if panel is not None:
#         cv2.imshow(title, panel)



# Visualizer(sd, "StepDiff").show_all()
# Visualizer(dv,   "Derivative").show_all()
# Visualizer(sd, "StepDiff").export_csv("stepdiff.csv")


# #RT video
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Camera not accessible")
#     exit()
    
# # Skip frame setup todo

# while True:
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     #gray = cv2.resize(gray, (640, 360))

#     dv = Derivative(gray, DV_THRESH, DV_STEP, DV_BLUR, DV_SLOT, DV_SLOT_THRESHOLD)
#     dc = DenseCorners(dv)

#     edges   = dv.draw()
#     corners = edges.copy()
#     if dc.corners is not None:
#         for pt in dc.corners:
#             cv2.circle(corners, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), -1)
#             cv2.circle(corners, (int(pt[0]), int(pt[1])), 5, (0,   0,   0),    2)

#     cv2.imshow("Original", frame)
#     cv2.imshow("Edges", edges)
#     cv2.imshow("Corners", corners)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
