import cv2

# 0 integrated, 1 usb camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, 50, 100)

    cv2.imshow("Basic", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blur", blur)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()