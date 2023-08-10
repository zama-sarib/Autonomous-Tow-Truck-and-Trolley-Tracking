from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import imutils

model = YOLO('yolov8s.pt')

# results = model.predict(source=r"C:\Users\zama.sarib\Downloads\MS DHONI HELICOPTERS SHOTS _ MSD Hard Hitting _ MSD Whatsapp Status.mp4",show=True)
# print(results)
# cap = cv2.VideoCapture(r"C:\Users\zama.sarib\Downloads\MS DHONI HELICOPTERS SHOTS _ MSD Hard Hitting _ MSD Whatsapp Status.mp4")
# r"C:\Users\zama.sarib\Downloads\Car Dash.mp4"
cap = cv2.VideoCapture(r"C:\Users\zama.sarib\Downloads\Car Dash.mp4")
while True:
    # ret,image = cap.read()
    # # if ret:
    # # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # prediction = model.predict(image,show=True,conf=0.8,iou=0.7)
    # print(prediction)
    # # cv2.imshow('output',prediction)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break
    success, frame = cap.read()
    
    if success:
        frame = imutils.resize(frame, width=416, height=416)
        results = model(frame,conf=0.7,iou=0.6,classes=[0])
        annotated_frame = results[0].plot()
        
        for r in results:
            for boxes in r.boxes.xyxy:
                x1,y1,x2,y2 = boxes[0].item(),boxes[1].item(),boxes[2].item(),boxes[3].item()
                mid_x = ((y1/384+y2/384)/2)
                mid_y = ((x1/640+x2/640)/2)
                apx_distance = round(((1 - (y2/640 - y1/640)))**3,1)
                print(mid_x,mid_y,apx_distance)
                if apx_distance <= 0.8:
                    if mid_x >= 0.2 and mid_x <= 0.7:
                        # print("Inside **********************************************************")
                    # img = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                        cv2.putText(annotated_frame, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            


        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break