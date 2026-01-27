import cv2
 
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    cv2.imshow("Video", frame)
    cv2.imshow("Grayscale video", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()