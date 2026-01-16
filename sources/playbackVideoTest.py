import cv2
 
cap = cv2.VideoCapture("sources/video/mov_video.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    cv2.imshow("Video", frame)
 
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()