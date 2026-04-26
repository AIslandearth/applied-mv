import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from collections import deque
from detectionAndCalcTools import *

# Tennis ball diameter in meters, ~67mm
BALL_DIAMETER = 0.067

# hsvMin = np.array([25, 100, 100])
# hsvMax = np.array([45, 255, 255])

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

maxSpd = 0
spdKmh = 0
prevPos = None
prevTime = None
prevSpd = 0
radiuses = []
paused = False
speedLog = []
frame = None

def processFrame(cap, frame):
    global prevPos, prevTime, prevSpd, maxSpd

    spdKmh, prevPos, prevTime, prevSpd, timeStamp = processAndVisualizeObject(
                                                    cap, frame,
                                                    THRESHOLD, STEP, BALL_DIAMETER,
                                                    hsvValues, HSV_THRESH,
                                                    prevPos, prevSpd, prevTime, radiuses, fps
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
                prevPos, prevTime, prevSpd = None, None, 0
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








#
#
# FUNCTIONING ONE BUT WITHOUT PAUSING
#
#

# while cap.isOpened():
#     if not paused:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         timeStamp = frameNum / fps

#         ball = detectBall(frame, hsvMin, hsvMax)

#         spd_kmh = None
#         acc = None

#         if ball:
#             x, y, r = ball
#             # Keep max 30 ball dimensions for calc average, automatically drop the tail
#             radiuses = deque(maxlen=30)
            
#             radiuses.append(r)
#             avgRadius = sum(radiuses) / len(radiuses)
#             pxPerMeter = (avgRadius * 2) / BALL_DIAMETER_M

#             if prevPos and prevTime:
#                 px, py = prevPos
#                 dt = timeStamp - prevTime

#                 if dt > 0:
#                     dist_m = math.sqrt((x - px)**2 + (y - py)**2) / pxPerMeter
#                     spd_ms = dist_m / dt
#                     spd_kmh = spd_ms * 3.6
#                     # acceleration meters per seconds powered to two
#                     acc = (spd_ms - prevSpd) / dt
#                     prevSpd = spd_ms

#             prevPos  = (x, y)
#             prevTime = timeStamp

#         drawBall(frame, ball, spd_kmh, acc)
#         cv2.imshow("Speed and acceleration detector", frame)
        
#     key = cv2.waitKey(int(waitTime_ms / 2)) & 0xFF

#     if key == ord('q'):
#         break
#     elif key == ord(' '):
#         paused = not paused
#     elif key == ord('d') and paused:
#         ret, frame = cap.read()
#         if ret:
#             cv2.imshow("Speed and acceleration detector", frame)
#     elif key == ord('a') and paused:
#         frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         # -2 to return to previous loop's frame, not current one as proceeded by index already at above
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum - 2)
#         ret, frame = cap.read()
#         if ret:
#             cv2.imshow("Speed and acceleration detector", frame)

# cap.release()
# cv2.destroyAllWindows()


#
#
# LOOP SCAN AND FPS TESTS
#
#

# if not cap.isOpened():
#     print("Video file not found")
#     exit()

# # Find fps and frame count for proper playback speed
# fps = cap.get(cv2.CAP_PROP_FPS)
# # Count total amount of frames in video
# #frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# # Time to wait per frame
# waitTime_ms = (1000 / fps)

# print(fps)
# print(waitTime_ms)
# #start = 0
# #elapsedTotal = 0

# frameIndex = 0
# while cap.isOpened():
#     #start = time.perf_counter()
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Read only every other frame for 60fps -> 30fps as the pc jitter / opencv scan time seems to be the braking factor
#     if (frameIndex % 2 == 0):
#         cv2.imshow("Video", frame)
#     frameIndex += 1

#     # "Set" playback speed via 1000ms / 2 as even the waitTime is calc for 60fps the jitter/scan time messes it
#     if cv2.waitKey(int(waitTime_ms / 2)) & 0xFF == ord('q'):
#         break
    
#     #cycleTime = time.perf_counter() - start
#     #elapsedTotal += cycleTime
#     #print(cycleTime)
#     #print(elapsedTotal)

# cap.release()
# cv2.destroyAllWindows()