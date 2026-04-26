import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

yoloModel = YOLO("yolo26n.pt")

#yoloModel.track(source="sources/video/dragSnapShot.mp4")

cap = cv2.VideoCapture("sources/video/dragSnapShot.mp4")

if not cap.isOpened():
    print("File not found")

names = yoloModel.names

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = yoloModel(frame, conf=0.25)
    # person = 0, sportsball = 32
    r = results[0]
    
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            clsId = int(box.cls[0])
            label = f"{names.get(clsId, clsId)} {conf:.2f}"
            
            cv2.rectangle(frame, (int(x1), int(y1), int(x2), int(y2)), color=(0, 0, 255), thickness=1)
            cv2.putText(frame, label, (int(x1), int(y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            
    cv2.imshow("Test yolo n -model", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()