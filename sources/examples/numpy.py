import numpy as np
import cv2

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
 
print(a + b)   # [5 7 9]
print(a * 2)   # [2 4 6]

m = np.array([[1, 2], [3, 4]])
print(m.T)     # transpose

# Transpose long
#a = np.array([[1, 2, 3],
#              [4, 5, 6],
#              [7, 8, 9]])
#print(np.transpose(a))

#Output
#[[1 4 7]
# [2 5 8]
# [3 6 9]]

# Parameters
# cv2.IMREAD_COLOR (default)
# cv2.IMREAD_GRAYSCALE
# cv2.IMREAD_UNCHANGED

img = cv2.imread("image.jpg", "something") # something as flags
gray = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found or could not be loaded")
print(type(img))
print(img.shape)

cv2.imwrite("output.jpg", img)

key = cv2.waitKey(0) # delay 0ms
print(key)

## Matplotlib for graphics viewing
#import matplotlib.pyplot as plt
 
#img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.imshow(img_rgb)
#plt.axis("off")
#plt.show()