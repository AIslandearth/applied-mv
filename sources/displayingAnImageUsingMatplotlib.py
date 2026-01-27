import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sources/img/testImage.png")

if img is None:
    print("Couldn't read the image")
    exit()

# BGR to RGB for Matplot
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(imgRGB)
plt.axis("off")
plt.title("RGB image")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()