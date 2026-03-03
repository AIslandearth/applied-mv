#              #
# Subfunctions #
#              #

# Grayscale
def to_grayscale(image):
    height = len(image)
    width = len(image[0])
    gray = [[0] * width for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            b, g, r = image[y][x]
            # Luminance formula — same as cv2.COLOR_BGR2GRAY
            gray[y][x] = int(0.114 * b + 0.587 * g + 0.299 * r)
    
    return gray

# Gaussian blur
def gaussian_blur(gray):
    height = len(gray)
    width = len(gray[0])
    blurred = [[0] * width for _ in range(height)]
    
    # 3x3
    kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
    kernel_sum = 16

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            total = 0
            for ky in range(3):
                for kx in range(3):
                    pixel = gray[y + ky - 1][x + kx - 1]
                    total += pixel * kernel[ky][kx]
            blurred[y][x] = total // kernel_sum
    
    return blurred

# Threshold
def threshold(gray, thresh=127):
    height = len(gray)
    width = len(gray[0])
    binary = [[0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if gray[y][x] < thresh:
                binary[y][x] = 1   # e.g dark = foreground
            else:
                binary[y][x] = 0   # e.g light = background

    return binary

# Gradient edge detection
def find_edges(gray):
    height = len(gray)
    width = len(gray[0])
    edges = [[0] * width for _ in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Sobel X and Y
            gx = (gray[y-1][x+1] + 2*gray[y][x+1] + gray[y+1][x+1]) - \
                 (gray[y-1][x-1] + 2*gray[y][x-1] + gray[y+1][x-1])

            gy = (gray[y+1][x-1] + 2*gray[y+1][x] + gray[y+1][x+1]) - \
                 (gray[y-1][x-1] + 2*gray[y-1][x] + gray[y-1][x+1])

            magnitude = int((gx**2 + gy**2) ** 0.5)

            if magnitude > 127:
                edges[y][x] = 1
            else:
                edges[y][x] = 0

    return edges

# Line detection
def find_lines(gray, y):
    width = len(gray[0])
    
    on_edge = False
    edge_start = -1
    line_centers = []

    i = 0
    while i < width:
        pixel = gray[y][i]

        if pixel < 127:
            if not on_edge:
                edge_start = i
                on_edge = True
        else:
            if on_edge:
                edge_center = (edge_start + i) // 2
                line_centers.append(edge_center)
                on_edge = False
        i += 1

    return line_centers

# Find shape corners
def find_corners(contour_points):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    top_left     = None
    top_right    = None
    bottom_left  = None
    bottom_right = None

    for y, x in contour_points:
        if x + y < min_x + min_y:
            top_left = (y, x)       # smallest sum
        if x + y > max_x + max_y:
            bottom_right = (y, x)   # largest sum
        if x - y > max_x - max_y:
            top_right = (y, x)      # largest diff
        if x - y < min_x - min_y:
            bottom_left = (y, x)    # smallest diff

        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)

    return top_left, top_right, bottom_right, bottom_left

# Find contour
def find_contour(binary):
    height = len(binary)
    width = len(binary[0])
    contour = []

    # Simple boundary scan, find transitions
    for y in range(height):
        for x in range(width):
            if binary[y][x] == 1:
                # Check if any neighbor is background
                is_boundary = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if binary[ny][nx] == 0:
                                is_boundary = True
                if is_boundary:
                    contour.append((y, x))

    return contour

#          #
# Combined #
#          #
def process_grid(image):
    gray    = to_grayscale(image)
    blurred = gaussian_blur(gray)
    binary  = threshold(blurred)
    edges   = find_edges(blurred)
    contour = find_contour(binary)
    corners = find_corners(contour)
    
    y = len(gray) // 2
    lines   = find_lines(gray, y)
    
    return corners, lines


#                         #
# Canny subfunctions e.g. #
#                         #

def non_max_suppression(magnitude, direction):
    height = len(magnitude)
    width = len(magnitude[0])
    result = [[0] * width for _ in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle = direction[y][x] % 180  # normalize to 0-180

            # Compare with neighbors along gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                n1 = magnitude[y][x+1]   # horizontal neighbors
                n2 = magnitude[y][x-1]
            elif 22.5 <= angle < 67.5:
                n1 = magnitude[y-1][x+1] # diagonal neighbors
                n2 = magnitude[y+1][x-1]
            elif 67.5 <= angle < 112.5:
                n1 = magnitude[y-1][x]   # vertical neighbors
                n2 = magnitude[y+1][x]
            else:
                n1 = magnitude[y+1][x+1] # other diagonal
                n2 = magnitude[y-1][x-1]

            # Keep only if local maximum
            if magnitude[y][x] >= n1 and magnitude[y][x] >= n2:
                result[y][x] = magnitude[y][x]
            else:
                result[y][x] = 0

    return result

# 3 option threshold
def double_threshold(magnitude, low=50, high=150):
    height = len(magnitude)
    width = len(magnitude[0])
    
    STRONG = 255
    WEAK   = 50
    NONE   = 0
    
    result = [[NONE] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            val = magnitude[y][x]
            if val >= high:
                result[y][x] = STRONG    # definitely an edge
            elif val >= low:
                result[y][x] = WEAK      # maybe an edge
            else:
                result[y][x] = NONE      # not an edge

    return result

# Define whether to connect weak edges or not ("DBSCAN")
def hysteresis(thresholded):
    height = len(thresholded)
    width = len(thresholded[0])
    result = [[0] * width for _ in range(height)]

    STRONG = 255
    WEAK   = 50

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if thresholded[y][x] == STRONG:
                result[y][x] = STRONG    # keep strong edges
            elif thresholded[y][x] == WEAK:
                # Keep weak edge only if connected to strong neighbor
                connected = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if thresholded[y+dy][x+dx] == STRONG:
                            connected = True
                if connected:
                    result[y][x] = STRONG
                else:
                    result[y][x] = 0     # discard isolated weak edge

    return result

#                 #
# Full canny e.g. #
#                 #
import math

def canny(image, low=50, high=150):
    # Step 1 — grayscale
    gray = to_grayscale(image)

    # Step 2 — blur
    blurred = gaussian_blur(gray)

    # Step 3 — gradients (Sobel)
    height = len(blurred)
    width  = len(blurred[0])
    
    magnitude = [[0] * width for _ in range(height)]
    direction = [[0] * width for _ in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = (blurred[y-1][x+1] + 2*blurred[y][x+1] + blurred[y+1][x+1]) - \
                 (blurred[y-1][x-1] + 2*blurred[y][x-1] + blurred[y+1][x-1])

            gy = (blurred[y+1][x-1] + 2*blurred[y+1][x] + blurred[y+1][x+1]) - \
                 (blurred[y-1][x-1] + 2*blurred[y-1][x] + blurred[y-1][x+1])

            magnitude[y][x] = int(math.sqrt(gx**2 + gy**2))
            direction[y][x] = math.degrees(math.atan2(gy, gx))

    # Step 4 — non maximum suppression
    suppressed = non_max_suppression(magnitude, direction)

    # Step 5 — double threshold
    thresholded = double_threshold(suppressed, low, high)

    # Step 6 — hysteresis
    edges = hysteresis(thresholded)

    return edges

# Perspective warp
def perspective_warp(image, src, dst, output_size):
    # Build transformation matrix from 4 point correspondences
    # src = 4 corners in original image
    # dst = where they should map to
    
    def build_matrix(src, dst):
        # Solve Ax = b for homography matrix H
        A = []
        b = []
        for i in range(4):
            xs, ys = src[i]
            xd, yd = dst[i]
            A.append([-xs, -ys, -1, 0, 0, 0, xd*xs, xd*ys])
            A.append([0, 0, 0, -xs, -ys, -1, yd*xs, yd*ys])
            b.append(-xd)
            b.append(-yd)
        return A, b

    def solve(A, b):
        # Gaussian elimination
        n = len(b)
        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col+1, n):
                if abs(A[row][col]) > abs(A[max_row][col]):
                    max_row = row
            A[col], A[max_row] = A[max_row], A[col]
            b[col], b[max_row] = b[max_row], b[col]

            for row in range(col+1, n):
                if A[col][col] == 0:
                    continue
                factor = A[row][col] / A[col][col]
                for k in range(col, n):
                    A[row][k] -= factor * A[col][k]
                b[row] -= factor * b[col]

        # Back substitution
        x = [0] * n
        for i in range(n-1, -1, -1):
            x[i] = b[i]
            for j in range(i+1, n):
                x[i] -= A[i][j] * x[j]
            x[i] /= A[i][i]
        return x

    A, b = build_matrix(src, dst)
    h = solve(A, b)

    # Homography matrix
    H = [
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0 ]
    ]

    out_h, out_w = output_size
    output = [[0] * out_w for _ in range(out_h)]

    # Map each output pixel back to source (inverse warp)
    for y in range(out_h):
        for x in range(out_w):
            # Apply inverse homography
            denom = H[2][0]*x + H[2][1]*y + H[2][2]
            src_x = (H[0][0]*x + H[0][1]*y + H[0][2]) / denom
            src_y = (H[1][0]*x + H[1][1]*y + H[1][2]) / denom

            # Bilinear interpolation
            x0, y0 = int(src_x), int(src_y)
            x1, y1 = x0 + 1, y0 + 1

            if 0 <= x0 < len(image[0])-1 and 0 <= y0 < len(image)-1:
                # Fractional parts
                dx = src_x - x0
                dy = src_y - y0

                # Weighted average of 4 neighboring pixels
                top    = image[y0][x0] * (1-dx) + image[y0][x1] * dx
                bottom = image[y1][x0] * (1-dx) + image[y1][x1] * dx
                output[y][x] = int(top * (1-dy) + bottom * dy)

    return output

#             #
# Ransac e.g. #
#             #
import math
import random

def fit_line(p1, p2):
    # Line equation: ax + by + c = 0
    x1, y1 = p1
    x2, y2 = p2
    
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    
    return a, b, c

def point_to_line_distance(point, line):
    x, y = point
    a, b, c = line
    
    denom = math.sqrt(a**2 + b**2)
    if denom == 0:
        return float('inf')
    
    return abs(a*x + b*y + c) / denom

def ransac_line(points, iterations=1000, threshold=2.0):
    best_line      = None
    best_inliers   = []
    best_count     = 0

    i = 0
    while i < iterations:
        # Pick 2 random points
        p1, p2 = random.sample(points, 2)

        # Fit line to those 2 points
        line = fit_line(p1, p2)

        # Count how many points agree (inliers)
        inliers = []
        for point in points:
            dist = point_to_line_distance(point, line)
            if dist < threshold:
                inliers.append(point)

        # Keep best result
        if len(inliers) > best_count:
            best_count   = len(inliers)
            best_inliers = inliers
            best_line    = line

        i += 1

    return best_line, best_inliers

#                   #
# Ransac usage e.g. #
#                   #
def find_all_lines(edge_points, num_lines=10):
    remaining_points = edge_points.copy()
    lines = []

    for _ in range(num_lines):
        if len(remaining_points) < 2:
            break

        # Find best line in remaining points
        line, inliers = ransac_line(remaining_points, iterations=500)

        if len(inliers) < 10:  # minimum points to be a real line
            break

        lines.append((line, inliers))

        # Remove inliers — look for next line
        remaining_points = [p for p in remaining_points if p not in inliers]

    return lines

#                        #
# Warp n perspective fix #
#                        #

# Manual low level approach
import math

def solve_homography(src, dst):
    # Build system of equations
    # Each point gives 2 equations
    # 4 points = 8 equations = solve for 8 unknowns in H matrix
    A = []
    b = []

    for i in range(4):
        xs, ys = src[i]
        xd, yd = dst[i]

        A.append([-xs, -ys, -1,  0,   0,   0,  xd*xs, xd*ys])
        A.append([ 0,   0,   0, -xs, -ys, -1,  yd*xs, yd*ys])
        b.append(-xd)
        b.append(-yd)

    # Gaussian elimination to solve Ax = b
    n = len(b)
    for col in range(n):
        # Find pivot row
        max_row = col
        for row in range(col+1, n):
            if abs(A[row][col]) > abs(A[max_row][col]):
                max_row = row
        A[col], A[max_row] = A[max_row], A[col]
        b[col], b[max_row] = b[max_row], b[col]

        for row in range(col+1, n):
            if A[col][col] == 0:
                continue
            factor = A[row][col] / A[col][col]
            for k in range(col, n):
                A[row][k] -= factor * A[col][k]
            b[row] -= factor * b[col]

    # Back substitution
    h = [0.0] * n
    for i in range(n-1, -1, -1):
        h[i] = b[i]
        for j in range(i+1, n):
            h[i] -= A[i][j] * h[j]
        h[i] /= A[i][i]

    # Build 3x3 homography matrix
    H = [
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0 ]
    ]
    return H


def warp_manual(gray, corners, output_size=500):
    src = corners
    dst = [
        [0,          0         ],
        [output_size, 0         ],
        [output_size, output_size],
        [0,          output_size]
    ]

    H = solve_homography(src, dst)

    # Create empty output image
    output = [[0] * output_size for _ in range(output_size)]

    # For each output pixel find where it comes from in source
    for y in range(output_size):
        for x in range(output_size):

            # Apply inverse homography
            denom = H[2][0]*x + H[2][1]*y + H[2][2]
            src_x = (H[0][0]*x + H[0][1]*y + H[0][2]) / denom
            src_y = (H[1][0]*x + H[1][1]*y + H[1][2]) / denom

            # Bilinear interpolation — smooth pixel sampling
            x0 = int(src_x)
            y0 = int(src_y)
            x1 = x0 + 1
            y1 = y0 + 1

            if 0 <= x0 < len(gray[0])-1 and 0 <= y0 < len(gray)-1:
                dx = src_x - x0
                dy = src_y - y0

                # Weighted average of 4 neighbors
                top    = gray[y0][x0] * (1-dx) + gray[y0][x1] * dx
                bottom = gray[y1][x0] * (1-dx) + gray[y1][x1] * dx
                output[y][x] = int(top * (1-dy) + bottom * dy)

    return output

# OpenCV warp, needs corners to find first
import cv2
import numpy as np

def warp_opencv(image, corners):
    # corners = [top_left, top_right, bottom_right, bottom_left]
    src = np.float32(corners)
    dst = np.float32([
        [0,   0  ],
        [500, 0  ],
        [500, 500],
        [0,   500]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (500, 500))
    return warped

# Usage
corners = [
    [47,  23 ],   # top left
    [412, 18 ],   # top right
    [430, 445],   # bottom right
    [31,  440]    # bottom left
]
warped = warp_opencv(gray, corners)
