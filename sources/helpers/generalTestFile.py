#import os
#import cv2

#from helpers import videoConverter
#from video import *

# If simple video file conversion, try from PowerShell: ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -an output.mp4

#def main():
#    video_in = ""   ######
#    video_out = videoConverter.convert(video_in)
#    print("Converted to:", video_out)
#
    #path = r"../video/test_video.MP4"
    #print(os.path.exists(path))
#
    #print("OpenCV:", cv2.__version__)
    #print("OpenCV path:", cv2.__file__)
#
#if __name__ == "__main__":
#    main()