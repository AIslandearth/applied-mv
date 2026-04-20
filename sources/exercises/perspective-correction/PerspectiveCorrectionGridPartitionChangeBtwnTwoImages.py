# Use these points
top_left = (np.float64(181.78094550703716), np.float64(85.94911584265608)) 
top_right = (np.float64(514.5795423214778), np.float64(76.44058450510063)) 
bottom_left = (np.float64(68.59891703732352), np.float64(398.17540127634885)) 
bottom_right = (np.float64(635.2059086839749), np.float64(389.4583706356311))




prev_cells = extractCells(warped, gridPoints)

# next frame...
curr_cells = extractCells(warped, gridPoints)

for i, row in enumerate(curr_cells):
    for j, cell in enumerate(row):
        if detectChange(prev_cells[i][j], cell):
            print(f"Change detected at cell [{i}][{j}]")

# assigment description

# Learning Objectives

# By completing this assignment, students will be able to:

# Correct perspective in two given images using provided corner points.
# Partition a rectified image into a 9 × 10 grid of equally sized regions.
# Compare two rectified images and determine which regions have changed.
# Visualize and report the detected changes.
# Input Data
# You are given two images taken of the same scene from (approximately) the same viewpoint but at different times. For each image, use the following corner points for perspective correction:

# top_left = (np.float64(181.78094550703716), np.float64(85.94911584265608)) 
# top_right = (np.float64(514.5795423214778), np.float64(76.44058450510063)) 
# bottom_left = (np.float64(68.59891703732352), np.float64(398.17540127634885)) 
# bottom_right = (np.float64(635.2059086839749), np.float64(389.4583706356311))
# Use the same point ordering for both images and ensure that both rectified outputs are the same size.
 

# 1) Perspective Correction
# Load both images.
# Arrange source points in the order:
# top-left, top-right, bottom-right, bottom-left.
# Define a destination rectangle (width, height) and map the source quadrilateral to this rectangle using:
# cv2.getPerspectiveTransform(src_pts, dst_pts)
# cv2.warpPerspective(image, M, (width, height))
# Produce rectified versions of both images with identical output size.
# Tell in your video report:

# Why the point order matters.
# How you chose the output width and height (e.g., using distances between points).
 

# 2) Split the Rectified Image into a 9 × 10 Grid
# Compute cell sizes:
# cell_w = width / 10
# cell_h = height / 9
# Extract each grid cell using NumPy slicing: img[y1:y2, x1:x2].
# (Optional but recommended) Visualize the grid by drawing lines or overlays on the image.
 

# 3) Detect Changes Between the Two Images
# For each corresponding grid cell in rectified image A and rectified image B:

# Compute a difference score using one of the following methods:
# Absolute difference + mean: cv2.absdiff then np.mean(...)
# MSE (Mean Squared Error)
# SSIM (if external library is allowed, e.g., skimage.metrics.structural_similarity)
# Choose a threshold to classify a cell as “changed”. For example:
# A cell is considered changed if the mean absolute difference > 20.
# You may adjust the threshold based on data characteristics.
# Tell in your video report:

# A list of changed cells as (row, col) indices (rows: 0–8, cols: 0–9).
# A visualization (e.g., red rectangles) highlighting changed cells on the rectified image.