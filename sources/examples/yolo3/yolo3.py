import cv2
import time
from ultralytics import YOLO
 
model = YOLO("yolo26n.pt")
pose_model = YOLO("yolo26n-pose.pt") 
names = model.names
detected_boxes = []
 
cap = cv2.VideoCapture("potku.mp4")
 
if not cap.isOpened():
    print("Video not accessible")
    exit()
 
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
 
# store the most recently read frame for saving
last_frame = None
 
fps = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS:", fps)


 
def show_frame(frame_index):
    global last_frame
    
    start = time.perf_counter_ns()
    detected_boxes.clear()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        results = model(frame, conf=0.25, imgsz=640, verbose=False)
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
 
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                            color=(0, 255, 0), thickness=2)
                cv2.putText(frame, label, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)
                
                detected_boxes.append({
                            "bbox": (x1, y1, x2, y2),
                            "cls": cls_id,
                            "conf": conf,
                            "label": label
                        })
 
            
        cv2.imshow("frame", frame)
        # remember this frame in case user wants to save it
        
        last_frame = frame
        
    end = time.perf_counter_ns()
    elapsed_us = (end - start) / 1000
    print(f"elapsed time: {elapsed_us:.2f} µs")
 
def on_mouse(event, x, y, flags, param):
    """Mouse callback for left/right clicks."""
    global current_frame, last_frame
    
    
    
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            # Rullaus YLÖS = eteenpäin
            if current_frame < total_frames - 1:
                current_frame += 1
                show_frame(current_frame)
        else:
            # Rullaus ALAS = taaksepäin
            if current_frame > 0:
                current_frame -= 1
                show_frame(current_frame)
 
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")
        for box in detected_boxes:
            x1, y1, x2, y2 = box["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"Clicked on detected object: {box['label']}")
                crop = last_frame[int(y1):int(y2), int(x1):int(x2)]
                results = pose_model(crop, conf=0.25, imgsz=640)
                annotated = results[0].plot()
                cv2.imshow("YOLO26 Pose", annotated)
                    
                #cv2.imshow("cropped", crop)
                break
 
    #if event == cv2.EVENT_LBUTTONDOWN:
    #    if current_frame > 0:
    #        current_frame -= 1
    #        show_frame(current_frame)
 
    # OIKEA nappi = eteenpäin
    #if event == cv2.EVENT_RBUTTONDOWN:
    #    if current_frame < total_frames - 1:
    #        current_frame += 1
    #        show_frame(current_frame)
 
# Setup
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", on_mouse)
 
# Show first frame
show_frame(current_frame)
 
# Main loop
while True:
    key = cv2.waitKey(20)
 
    if key == ord('s'):
        # save current frame as PNG
        if last_frame is not None:
            filename = f"frame_{current_frame:06d}.png"
            cv2.imwrite(filename, last_frame)
            print(f"Saved {filename}")
        else:
            print("No frame available to save")
    elif key == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()