import cv2
import numpy as np

img = cv2.imread("sources/img/testImage.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Couldn't read the image")
    exit()

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
