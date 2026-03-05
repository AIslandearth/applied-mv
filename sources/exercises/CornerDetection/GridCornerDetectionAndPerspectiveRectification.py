import cv2
from detectors import StepDiff, Derivative, SimpleCorners, DenseCorners, warp_to_corners
from visualize import Visualizer

img = cv2.imread("sources/img/image1_1.png")
if img is None:
    print("Cannot read image")
    exit

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Draw dot radius
CIRCLE_R  = 3

# Step based difference
SD_THRESH = 30
SD_STEP   = 3

# Derivative based difference
DV_THRESH = 10
DV_STEP   = 7
DV_BLUR   = 5

# StepDiff + DenseCorners
sd = StepDiff(gray, threshold=SD_THRESH, step=SD_STEP)
sd.detect()
#corners_sd = DenseCorners(sd.ys, sd.centers, gray.shape, slots=4, slot_threshold=20).find()
corners_sd = SimpleCorners(sd.ys, sd.centers).find()

a1 = sd.draw(circle_r=CIRCLE_R)
a1 = sd.draw_corners(a1, corners_sd)

a2 = None
warp_sd = warp_to_corners(gray, corners_sd)
if warp_sd is not None:
    warped_sd, _ = warp_sd
    sd2 = StepDiff(warped_sd, threshold=SD_THRESH, step=SD_STEP)
    sd2.detect()
    #corners_sd2 = DenseCorners(sd2.ys, sd2.centers, warped_sd.shape, slots=4, slot_threshold=20).find()
    corners_sd2 = SimpleCorners(sd.ys, sd.centers).find()
    a2 = sd2.draw(circle_r=CIRCLE_R)
    a2 = sd2.draw_corners(a2, corners_sd2)

# Derivative + SimpleCorners
dv = Derivative(gray, threshold=DV_THRESH, step=DV_STEP, blur=DV_BLUR)
dv.detect()
#corners_dv = SimpleCorners(dv.ys, dv.centers).find()
corners_dv = DenseCorners(dv.ys, dv.centers, gray.shape, slots=4, slot_threshold=20).find()

b1 = dv.draw(circle_r=CIRCLE_R)
b1 = dv.draw_corners(b1, corners_dv)

b2 = None
warp_dv = warp_to_corners(gray, corners_dv)
if warp_dv is not None:
    warped_dv, _ = warp_dv
    dv2 = Derivative(warped_dv, threshold=DV_THRESH, step=DV_STEP, blur=DV_BLUR)
    dv2.detect()
    #corners_dv2 = SimpleCorners(dv2.ys, dv2.centers).find()
    corners_dv2 = DenseCorners(dv.ys, dv.centers, gray.shape, slots=4, slot_threshold=20).find()
    b2 = dv2.draw(circle_r=CIRCLE_R)
    b2 = dv2.draw_corners(b2, corners_dv2)

cv2.imshow("StepDiff original", a1)
cv2.imshow("Derivative original", b1)
if a2 is not None:
    cv2.imshow("StepDiff rectified", a2)
if b2 is not None:
    cv2.imshow("Derivative rectified", b2)

cv2.waitKey(0)
cv2.destroyAllWindows()

vis_sd = Visualizer(sd, title="StepDiff")
vis_sd.show_all(row=gray.shape[0]//2, col=gray.shape[1]//2)
vis_sd.export_csv("stepdiff_points.csv")
vis_sd.export_excel("stepdiff_points.xlsx")

vis_dv = Visualizer(dv, title="Derivative")
vis_dv.show_all(row=gray.shape[0]//2, col=gray.shape[1]//2)
vis_dv.export_csv("derivative_points.csv")



# #import tkinter as tk
# #import itertools
# import cv2
# import numpy as np
# import csv
# from CSVWriter import CSVWriter
# from differenceKindOf import Diff

# img = cv2.imread("sources/img/image1_1.png")

# d = Diff(img, step=3, threshold=30).detect()

# draw1 = d.draw()
# cv2.imshow("Original with edges", draw1)

# warped = d.warp()
# if warped is not None:
#     d2 = Diff(warped, step=3, threshold=30).detect()
#     draw2 = d2.draw()
#     cv2.imshow("Warped with edges", draw2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #writer = CSVWriter(imgGray, "sources/exercises/cornerdetection/diffPixelValues.csv")

# # CSVwriter test
# # writer = CSVWriter("sources/img/image1_1.png", "sources/exercises/cornerdetection/pixelValues.csv")
# # #cv2.imshow("Original", imageOrig)
# # #cv2.imshow("Grayscale", writer.imageGray)
# # #cv2.imshow("Blur", writer.imageBlur)
# # #print(writer.imageBlur)

# cv2.waitKey(0)
# cv2.destroyAllWindows()