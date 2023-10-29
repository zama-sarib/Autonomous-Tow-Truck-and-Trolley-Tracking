import cv2
import torch
import numpy as np
import cv2 
import time 
from ultralytics import YOLO



# model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model = YOLO('yolov8n.pt')
device = 0 if torch.cuda.is_available else 'cpu'
print(torch.cuda.get_device_name())
print(device)

start_time = time.time()
# FPS update time in seconds
display_time = 2
fc = 0
FPS = 0

cap = cv2.VideoCapture("C:\\Users\\zamas\\Downloads\\Video\\RealSense Video Feed 11 July.mp4")
# cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model.to(device)(frame)
    annotated_frame = result[0].plot()

    for r in result:
        for bboxes in r.boxes.xyxy:
            # print(bboxes)
            x_mid = (bboxes[0] + bboxes[2]) / 2
            y_mid = (bboxes[1] + bboxes[3]) / 2
            cv2.circle(annotated_frame,(int(x_mid),int(y_mid)),2,(255, 0, 0),2)
    
    fc+=1
    TIME = time.time() - start_time

    if (TIME) >= display_time :
        FPS = fc / (TIME)
        fc = 0
        start_time = time.time()

    fps_disp = "FPS: "+str(FPS)[:5]
    cv2.putText(annotated_frame, fps_disp, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Inference", annotated_frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()