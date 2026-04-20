import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

MAX_FEATURES = 100

# Size of the display in use for img initial resizing
root = tk.Tk()
displayH = root.winfo_screenheight()
displayW = root.winfo_screenwidth()
# No need after display variables assigned
root.destroy()

images = []
images.append(cv2.cvtColor(cv2.imread("sources/img/crosswords.png"), cv2.COLOR_BGR2RGB))
images.append(cv2.cvtColor(cv2.imread("sources/img/crosswords2.png"), cv2.COLOR_BGR2RGB))

# images = [
#     cv2.imread("sources/img/crosswords2.png"),
#     cv2.imread("sources/img/crosswords.png"),
# ]

for i in range(len(images)):
    
    if images[i] is None:
        print("Couldn't read the image")
        exit()
    # img height and width, order = (height, width, channels)
    imgH, imgW = images[i].shape[:2]

    # Resize if image is larger than the display
    if imgW > displayW or imgH > displayH:
        # 0.9 = "safe" margin for the resized window
        # Calculate scaling factor for resizing using width and height + margin
        scaling = min(displayW / imgW, displayH / imgH) * 0.9
        # Resized img
        images[i] = cv2.resize(images[i], (int(imgW * scaling), int(imgH * scaling)))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(images[0])
axes[0, 0].set_title("crosswords")

axes[0, 1].imshow(images[1])
axes[0, 1].set_title("crosswords2")

orb = cv2.ORB_create(MAX_FEATURES)

# Second param "None" = mask
keypoints1, descriptors1 = orb.detectAndCompute(images[0], None)
keypoints2, descriptors2 = orb.detectAndCompute(images[1], None)

#imgDisplay1 = cv2.drawKeypoints(images[0], keypoints1, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#imgDisplay2 = cv2.drawKeypoints(images[1], keypoints1, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Converting to list for sorting as tuples are immutable objects.
matches = list(matcher.match(descriptors1, descriptors2, None))

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Discard matches based on percentage
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# imMatches = cv2.drawMatches(images[0], keypoints1, images[1], keypoints2, matches, None)

axes[1, 0].imshow(images[0])
if keypoints1:
    pts = cv2.KeyPoint_convert(keypoints1)
    axes[1, 0].scatter(pts[:, 0], pts[:, 1], c='yellow', s=10, zorder=5)
axes[1, 0].set_title("keypoints1")

axes[1, 1].imshow(images[1])
if keypoints2:
    pts = cv2.KeyPoint_convert(keypoints2)
    axes[1, 1].scatter(pts[:, 0], pts[:, 1], c='yellow', s=10, zorder=5)
axes[1, 1].set_title("keypoints2")

# axes[1, 2].imshow(matches)
# axes[1, 2].set_title("original iwth matches")

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()