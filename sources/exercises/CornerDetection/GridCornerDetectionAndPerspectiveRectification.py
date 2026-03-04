#import tkinter as tk
#import itertools
import cv2
import numpy as np
import csv
from CSVWriter import CSVWriter
from differenceKindOf import Diff

gray = cv2.imread("sources/img/image1_1.png", cv2.IMREAD_GRAYSCALE)

diff = Diff(
        gray,
        threshold = 20,    # adaptive diff sensitivity
        step           = 3,     # blur + lookup
        min_width      = 2,     # grid line min width
        max_width      = 20,    # grid line max width
    )
diff.detect()

cv2.imshow("Original with edges", diff.draw())

warped = diff.warp()
if warped is not None:
    diff_warped = Diff(warped,
                       threshold=30, step=2,
                       min_width=2,       max_width=30)
    diff_warped.detect()
cv2.imshow("Warped", warped)
cv2.imshow("Warped with edges", diff_warped.draw(warped))

#writer = CSVWriter(imgGray, "sources/exercises/cornerdetection/diffPixelValues.csv")

# CSVwriter test
# writer = CSVWriter("sources/img/image1_1.png", "sources/exercises/cornerdetection/pixelValues.csv")
# #cv2.imshow("Original", imageOrig)
# #cv2.imshow("Grayscale", writer.imageGray)
# #cv2.imshow("Blur", writer.imageBlur)
# #print(writer.imageBlur)

cv2.waitKey(0)
cv2.destroyAllWindows()