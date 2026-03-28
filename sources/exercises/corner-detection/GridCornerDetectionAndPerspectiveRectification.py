import sys
import cv2
from calcCorners_2 import *
from visualize import Visualizer
from findEdges_2 import *

# StepDiff
SD_THRESH = 30
SD_STEP   = 3
SD_SLOT = 10
SD_SLOT_THRESHOLD = 30

# Derivative
DV_THRESH = 60
DV_STEP   = 5
DV_BLUR   = 9
DV_SLOT = 1
DV_SLOT_THRESHOLD = 60

img  = cv2.imread("sources/img/image1_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#sd = StepDiff(gray, SD_THRESH, SD_STEP, SD_SLOT, SD_SLOT_THRESHOLD)
#dc = DenseCorners(sd)
#dv = Derivative(gray, DV_THRESH, DV_STEP, DV_BLUR, DV_SLOT, DV_SLOT_THRESHOLD)
#dc = SimpleCorners(dv)
dvv = Derivative(gray, DV_THRESH, DV_STEP, DV_BLUR, DV_SLOT, DV_SLOT_THRESHOLD)
dc = DenseCorners(dvv)
# Original + edges
edges = dvv.draw()
dc.draw_corners(edges)

# Warped
warped = warpToCorners(gray, dc.corners)
warped_edges = None

if warped is not None:
    #sd_warped = StepDiff(warped, SD_THRESH, SD_STEP, SD_SLOT, SD_SLOT_THRESHOLD)
    #dc_warped = DenseCorners(sd_warped)
    #warped_edges = sd_warped.draw()
    #dc_warped.draw_corners(warped_edges)
    
    dv_warped = Derivative(warped, DV_THRESH, DV_STEP, DV_BLUR, DV_SLOT, DV_SLOT_THRESHOLD)
    sc_warped = DenseCorners(dv_warped)
    warped_edges = dv_warped.draw()
    sc_warped.draw_corners(warped_edges)

# Display
panels = {
    "original":          cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
    "original + edges":  edges,
    "warped":            cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR) if warped is not None else None,
    "warped + edges":    warped_edges,
}

for title, panel in panels.items():
    if panel is not None:
        cv2.imshow(title, panel)

cv2.waitKey(0)
cv2.destroyAllWindows()

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
