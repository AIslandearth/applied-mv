import cv2
from matplotlib import pyplot as plt

img_BGR = cv2.imread('google-logo.jpg')
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
assert img_RGB is not None, "file could not be read, check with os.path.exists()"

blur = cv2.blur(img_RGB, (5, 5))

plt.subplot(121), plt.imshow(img_RGB), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
