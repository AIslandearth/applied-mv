import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('j.png', 0)
assert img is not None, "file could not be read, check with os.path.exists()"

kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)

plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
plt.xticks([]), plt.yticks([])

plt.show()
