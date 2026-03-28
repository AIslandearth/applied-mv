import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import math
import itertools
import own_tools
 
top_left = (np.float64(181.78094550703716), np.float64(85.94911584265608)) 
top_right = (np.float64(514.5795423214778), np.float64(76.44058450510063)) 
bottom_left = (np.float64(68.59891703732352), np.float64(398.17540127634885)) 
bottom_right = (np.float64(635.2059086839749), np.float64(389.4583706356311))
print(f"top_left:{top_left} top_right:{top_right} bottom_left:{bottom_left} bottom_right:{bottom_right}")
 
image = cv2.imread("challenges/images/image1_1.png")
#image = cv2.imread("challenges/images/image1_20260212_141741.png")
#image = cv2.imread("challenges/images/image2_20260212_141151.png")
#image_original = image.copy()
 
pts1 = np.float32([[int(top_left[0]), int(top_left[1])], [int(top_right[0]), int(top_right[1])], [int(bottom_left[0]), int(bottom_left[1])], [int(bottom_right[0]), int(bottom_right[1])]])
pts2 = np.float32([[0, 0], [900, 0], [0, 1000], [900, 1000]])
 
# Apply Perspective Transform Algorithm
matrix = cv2.getPerspectiveTransform(pts1, pts2)
perspective_corrected_image = cv2.warpPerspective(image, matrix, (900, 1000))
gray = cv2.cvtColor(perspective_corrected_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (15,15), 3)
 
#for y in range(50, 1000, 100):
#    cv2.line(perspective_corrected_image, (0,y),(899,y), (255,0,0))
    
    
#for x in range(50, 900, 100):
#    cv2.line(perspective_corrected_image, (x,0),(x,999), (255,0,0))
 
cropped_images = []
cropped_coordinates = [] 
for x in range(50, 900, 100):
    for y in range(50, 1000, 100):
        cv2.rectangle(perspective_corrected_image, (x-40,y-40), (x+40,y+40),(0,255,0))
        y1 = y - 40
        y2 = y + 40
        x1 = x - 40
        x2 = x + 40
        #crop = blurred[y-40:y+40, x-40:x+40]
        crop = blurred[y1:y2, x1:x2]
        cropped_images.append(crop)
        cropped_coordinates.append([(x1,y1),(x2,y2)])       
       
  
 
#image_new = cv2.imread("challenges/images/image1_2.png")
#cv2.imwrite("image_new.png", image_new)
#image_new = cv2.imread("challenges/images/image1_20260212_141418.png")
image_new = cv2.imread("challenges/images/image1_20260212_141815.png")
perspective_corrected_image_new = cv2.warpPerspective(image_new, matrix, (900, 1000))
gray_new = cv2.cvtColor(perspective_corrected_image_new, cv2.COLOR_BGR2GRAY)
blurred_new = cv2.GaussianBlur(gray_new, (15,15), 3) 
 
cropped_images_new = [] 
for x in range(50, 900, 100):
    for y in range(50, 1000, 100):
        cv2.rectangle(perspective_corrected_image_new, (x-40,y-40), (x+40,y+40),(0,255,0))
        crop = blurred_new[y-40:y+40, x-40:x+40]
        cropped_images_new.append(crop)
 
if(len(cropped_images) == len(cropped_images_new)):
    for crop_index in range(len(cropped_images)):
        
        A32 = cropped_images[crop_index].astype(np.float32)
        B32 = cropped_images_new[crop_index].astype(np.float32)
 
        mse = np.mean((A32 - B32) ** 2)
        mae = np.mean(np.abs(A32 - B32))
        print(f"crop_index: {crop_index} mse: {mse} mae: {mae}")
        
        if(mse > 100):
            cv2.imshow(f"difference_{crop_index}", cropped_images_new[crop_index]) 
            coordinates = cropped_coordinates[crop_index]
            point1 = coordinates[0]
            point2 = coordinates[1]
            cv2.rectangle(perspective_corrected_image,point1, point2, (255,255,255),10)
                   
        
    
 
       #print("MSE:", mse)
 
    
cv2.imshow("perspective_corrected_image", perspective_corrected_image)
cv2.imshow("perspective_corrected_image_new", perspective_corrected_image_new) 
 
#cv2.imshow("image", image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()