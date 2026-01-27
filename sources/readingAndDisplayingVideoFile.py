import cv2
 
cap = cv2.VideoCapture("sources/video/testVideo.MOV")

if not cap.isOpened():
    print("Video file not found")
    exit()

# Find fps for proper playback speed
fps = cap.get(cv2.CAP_PROP_FPS)

# 29.981... ->  ~ 30 fps
temp = ((fps * 100) + 5) / 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    cv2.imshow("Video", frame)
 
    # "Set" playback speed via rounded fps
    if cv2.waitKey(int(temp)) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()