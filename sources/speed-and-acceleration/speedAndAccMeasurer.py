# Atte Saarimaa 26.04.2026

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from collections import deque
from detectionAndCalculation import *

# Tennis ball diameter in meters, ~70mm
BALL_DIAMETER = 0.070
DEQUE_LENGTH = 5

hsvValues = np.array([40, 180, 180])
HSV_THRESH = 0.45 # +- Percentage for each channel

QUIT = ord('q')
KEY_SPACE = ord(' ')
KEY_D = ord('d')
KEY_A = ord('a')

THRESHOLD = 10 # Step difference required for edge detection
STEP = 2 # Index hop width

#TARGET_VALUE = 80 # Target color value of the img
#TARGET_THRESH = 0.13 # Percentage based +- hysteresis of the fixed gray value

cap = cv2.VideoCapture("sources/video/dragSnapShot.mp4")

if not cap.isOpened():
    print("Video file not found")
    exit()

# Find fps and total frame count for proper playback speed
fps = cap.get(cv2.CAP_PROP_FPS)
#frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# Time to wait per frame
waitTime_ms = (1000 / fps)

# Keep track of the last few detected ball pos, time and speed to filter/smooth the speed and acc calculation
radiuses = deque(maxlen=30)
posHistory = deque(maxlen=DEQUE_LENGTH)
timeHistory = deque(maxlen=DEQUE_LENGTH)
spdHistory = deque(maxlen=DEQUE_LENGTH)

maxSpd = 0
paused = False
speedLog = []
frame = None

def processFrame(cap, frame):
    global maxSpd

    spdKmh, acc , timeStamp = processAndVisualizeObject(
                                                    cap, frame,
                                                    THRESHOLD, STEP, BALL_DIAMETER,
                                                    hsvValues, HSV_THRESH,
                                                    fps, radiuses, posHistory, timeHistory, spdHistory
    )
    if spdKmh is not None and spdKmh > maxSpd:
        maxSpd = spdKmh
        speedLog.append((timeStamp, spdKmh))

while cap.isOpened():
    key = cv2.waitKey(int(waitTime_ms / 2)) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

    if paused:
        if key == ord('d'):
            ret, frame = cap.read()
            if ret:
                processFrame(cap, frame)
        elif key == ord('a'):
            frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum - 2)
            ret, frame = cap.read()
            if ret:
                clearHistory(posHistory, timeHistory, spdHistory)
                processFrame(cap, frame)
    else:
        ret, frame = cap.read()
        if not ret:
            break
        processFrame(cap, frame)

    if frame is not None:
        cv2.imshow("Speed and acceleration detector", frame)

cap.release()
cv2.destroyAllWindows()
print(f"Max speed: {maxSpd:.1f} km/h")

if speedLog:
    times  = [s[0] for s in speedLog]
    speeds = [s[1] for s in speedLog]
    maxIndex = speeds.index(max(speeds))
    
    fig, ax = plt.subplots()
    ax.plot(times, speeds)
    ax.plot(times[maxIndex], speeds[maxIndex], 'ro', markersize=8, label=f"Max: {maxSpd:.1f} km/h at {times[maxIndex]:.2f}s")
    ax.set(xlabel="Time (s)", ylabel="Speed (km/h)", title=f"Tennis ball speed — max {maxSpd:.1f} km/h")
    ax.legend()
    plt.show()