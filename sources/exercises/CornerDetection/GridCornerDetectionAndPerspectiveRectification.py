#import tkinter as tk
#import itertools
import cv2
import numpy as np
import csv
from CSVWriter import CSVWriter
from differenceKindOf import Diff

# imgGray = cv2.cvtColor("sources/img/image1_1.png", cv2.COLOR_BGR2GRAY)
imgGray = cv2.imread("sources/img/image1_1.png", cv2.IMREAD_GRAYSCALE)

diffArray = Diff(imgGray, 25, 5, 1, 60)
diffArray.detect()

imgEdges = diffArray.draw()
imgWarped = diffArray.warp()

cv2.imshow("Original", imgGray)
cv2.imshow("Original with edges", imgEdges)

if imgWarped is not None:
    # Detect again from reoriented picture
    diffArrayWarped = Diff(imgWarped, 25, 3, 1, 100)
    diffArrayWarped.detect()
    imgReoriented = diffArrayWarped.draw()
    
cv2.imshow("Warped", imgWarped)
cv2.imshow("Warped with edges", imgReoriented)

# # CSVwriter test
# #writer = CSVWriter("sources/img/image1_1.png", "sources/exercises/cornerdetection/pixelValues.csv")
# #cv2.imshow("Original", imageOrig)
# #cv2.imshow("Grayscale", writer.imageGray)
# #cv2.imshow("Blur", writer.imageBlur)
# #print(writer.imageBlur)

cv2.waitKey(0)
cv2.destroyAllWindows()