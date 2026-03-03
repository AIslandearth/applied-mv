import cv2
import numpy as np
import tkinter as tk
import itertools
import csv
from CSVWriter import CSVWriter

writer = CSVWriter("sources/img/image1_1.png", "sources/exercises/cornerdetection/pixelValues.csv")

#cv2.imshow("Original", imageOrig)
cv2.imshow("Grayscale", writer.imageGray)
cv2.imshow("Blur", writer.imageBlur)
#print(writer.imageBlur)

cv2.waitKey(0)
cv2.destroyAllWindows()