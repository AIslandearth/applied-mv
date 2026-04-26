import cv2
from ultralytics import YOLO
import time
 
# 1) Lataa YOLO26-malli (valitse koko: n/s/m/l/x)
model = YOLO("yolo26n.pt")  # kevyt ja nopea; muita: yolo26s.pt, yolo26m.pt, ...
 
# 2) Avaa videolähde:
#    - 0 = oletuswebkamera
#    - "video.mp4" = tiedosto
#    - "rtsp://user:pass@ip:554/..." = RTSP-kamera
# source = 0
cap = cv2.VideoCapture("test vid.mp4")
 
if not cap.isOpened():
    raise RuntimeError(f"Videolähdettä ei saatu auki")
 
# (Valinnainen) skaalaus suorituskykyyn: pienempi resoluutio -> nopeampi
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
# 3) Luokkien nimet mallista (COCO, jos käytät valmista .pt-mallia)
names = model.names  # dict: id -> class_name
 
# 4) Pääsilmukka
prev_t = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
 
    # 5) Inferenssi yhteen frameen
    #conf=0.25 on oletus; imgsz vaikuttaa nopeuteen/tarkkuuteen
    results = model(frame, conf=0.25) #, imgsz=640, verbose=False)
 
    # 6) Poimi tulokset: laatikot, luokat, scoret
    #    results on lista; tässä yksi frame -> results[0]
    r = results[0]
 
    # r.boxes sisältää tensorit: xyxy, conf, cls
    if r.boxes is not None:
        for box in r.boxes:
            # xyxy-koordinaatit (vasen-ylä & oikea-ala)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
 
            # 7) Piirrä laatikko + teksti
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)
 
    # 8) FPS-laskuri (valinnainen)
    now = time.time()
    fps = 1.0 / (now - prev_t)
    prev_t = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 50), 2)
 
    # 9) Näytä
    cv2.imshow("YOLO26 + OpenCV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 10) Siivous
cap.release()
cv2.destroyAllWindows()