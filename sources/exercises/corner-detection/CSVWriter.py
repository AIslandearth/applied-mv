import cv2
import numpy as np
import tkinter as tk
import itertools
import csv

class CSVWriter:
    
    def __init__(self, imgFile: str, csvFileName: str):
        self.imageOrig = imgFile
        self.csvFileName = csvFileName
    
        # If required to show original picture
        #imageOrig = cv2.imread("sources/img/image1_1.png")
        #self.imageGray = cv2.cvtColor((cv2.imread(self.imageOrig)), cv2.COLOR_BGR2GRAY)
        #self.imageBlur = cv2.GaussianBlur(self.imageGray, (15,15), 0)

        #self.height, self.width = self.imageGray.shape[:2]
        self.height, self.width = self.imageOrig.shape[:2]
        self.write_csv()

    def write_csv(self):

        # Open csv for writing
        with open(self.csvFileName, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=";")
    
            # Header
            writer.writerow([
                "Y",
                "X",
                "Gray",      
            ])
    
        # One row per X coordinate (horizontal)
        # If BGR or RBG then red, green, blue = image[y, x]
    
            for x in range(self.width):
            
                # Start test from middle in vertical dir
                y = int(self.height / 2)

                #gray = int(self.imageGray[y, x])
                gray = int(self.imageOrig[y, x])
                #blur = int(self.imageBlur[y, x])
                # BGR sum if want to calc avg when using color img
       
                writer.writerow([
                    y,
                    x,
                    gray
                ])