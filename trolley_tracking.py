import cv2
import torch
import numpy as np
import cv2 
import time 
from ultralytics import YOLO



model = YOLO('yolov8n.pt')
device = 0 if torch.cuda.is_available else 'cpu'
print(torch.cuda.get_device_name())
print(device)

start_time = time.time()
# FPS update time in seconds
display_time = 2
fc = 0
FPS = 0

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (416, 416))
    
    result = model.to(device)(frame)
    annotated_frame = result[0].plot()

    for r in result:
        for bboxes in r.boxes.xyxy:
            print(bboxes)
            if bboxes[0] > 10 and bboxes[2] < 400 and bboxes[1] > 10 and bboxes[3] < 400:
                x_mid = (bboxes[0] + bboxes[2]) / 2
                y_mid = (bboxes[1] + bboxes[3]) / 2
                cv2.circle(frame,(int(x_mid),int(y_mid)),2,(255, 0, 0),2)
                
                cv2.rectangle(frame,(int(bboxes[0]),int(bboxes[1])),(int(bboxes[2]),int(bboxes[3])),(255, 0, 0),2)
    
    fc+=1
    TIME = time.time() - start_time

    if (TIME) >= display_time :
        FPS = fc / (TIME)
        fc = 0
        start_time = time.time()

    fps_disp = "FPS: "+str(FPS)[:5]
    cv2.putText(frame, fps_disp, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.rectangle(frame,(10,10),(400,400),(0, 0, 255),2)
    cv2.imshow("YOLOv8 Inference", frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()