import cv2
import numpy as np

img = cv2.imread("sources/img/testImage.png")

if img is None:
    print("Couldn't read the image")
    exit()

cv2.imshow("Image", img)

imgGy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("sources/img/savedImage.png", imgGy)
cv2.imshow("Grayscale image", imgGy)

cv2.waitKey(0)
cv2.destroyAllWindows()