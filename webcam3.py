import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

from boxmot import DeepOCSORT


device = 0 if torch.cuda.is_available else 'cpu'
model = YOLO("./best.pt").to(device=device)
tracker = DeepOCSORT(
    model_weights=Path(r'C:\Users\zamas\Desktop\Autonomous-Vehicle-main\Autonomous-Vehicle-main\yolov5flask\yolo_tracking\examples\weights\resnet50_sarib.pt'),
    device='cuda:0',
    fp16=True,
)

vid = cv2.VideoCapture("C:\\Users\\zamas\\Downloads\\Video\\RealSense.mp4")
color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5
class_name = ['Trolley','person']

while True:
    ret, im = vid.read()
    frame = cv2.resize(im, (640, 480))
    results = model.to(device)(frame)
    annotated_frame = results[0].plot()

    # substitute by your object detector, input to tracker has to be N X (x, y, x, y, conf, cls)
    # dets = np.array([[144, 212, 578, 480, 0.82, 0],
    #                 [425, 281, 576, 472, 0.56, 65]])
    
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        probs = result.probs  # Class probabilities for classification outputs
        cls = boxes.cls.tolist()  # Convert tensor to list
        xyxy = boxes.xyxy
        conf = boxes.conf
        xywh = boxes.xywh  # box with xywh format, (N, 4)
            # for class_index in cls:
            #     class_name = class_names[int(class_index)]
                #print("Class:", class_name)

    pred_cls = np.array(cls)
    conf = conf.detach().cpu().numpy()
    xyxy = xyxy.detach().cpu().numpy()
    bboxes_xywh = xywh
    bboxes_xywh = xywh.cpu().numpy()
    bboxes_xywh = np.array(bboxes_xywh, dtype=float)

    # print(pred_cls)
    # print(xyxy)
    # print(conf)

    final = np.concatenate((xyxy,conf[:,None]),axis=1)
    dets = np.concatenate((final,pred_cls[:,None]),axis=1)
    
    # # tracks = tracker.update(bboxes_xywh, conf,pred_cls, frame)

    tracks = tracker.update(dets, annotated_frame) # --> (x, y, x, y, id, conf, cls, ind)

    xyxys = tracks[:, 0:4].astype('int') # float64 to int
    ids = tracks[:, 4].astype('int') # float64 to int
    confs = tracks[:, 5]
    clss = tracks[:, 6].astype('int') # float64 to int
    inds = tracks[:, 7].astype('int') # float64 to int

    # in case you have segmentations or poses alongside with your detections you can use
    # the ind variable in order to identify which track is associated to each seg or pose by:
    # segs = segs[inds]
    # poses = poses[inds]
    # you can then zip them together: zip(tracks, poses)

    # print bboxes with their associated id, cls and conf
    if tracks.shape[0] != 0:
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            # annotated_frame = cv2.rectangle(
            #     annotated_frame,
            #     (xyxy[0]+7, xyxy[1]),
            #     (xyxy[2], xyxy[3]),
            #     color,
            #     thickness
            # )
            cv2.putText(
                annotated_frame,
                f'id: {id}',
                (xyxy[0], xyxy[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                1
            )

    # show image with bboxes, ids, classes and confidences
    cv2.imshow('frame', annotated_frame)

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
