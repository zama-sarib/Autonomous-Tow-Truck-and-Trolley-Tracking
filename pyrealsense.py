"""
Inference yolov3 in Realsense D435 camera
Creator: Tony Do
Date: 9nd July, 2021
E-mail: vanhuong.robotics@gmail.com
- updated object depth information
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import sys

# Initialize the parameters


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] -1] for i in net.getUnconnectedOutLayers()]

def drawPredicted(classId, conf, left, top, right, bottom, frame,x ,y):
    cv2.rectangle(frame, (left,top), (right,bottom), (255,178,50),3)
    dpt_frame = pipeline.wait_for_frames().get_depth_frame().as_depth_frame()
    distance = dpt_frame.get_distance(x,y)
    cv2.circle(frame,(x,y),radius=1,color=(0,0,254), thickness=5)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s' %(classes[classId])
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label,(left,top-5), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,0),2)
    distance_string = "Dist: " + str(round(distance,2)) + " meter away"
    cv2.putText(frame,distance_string,(left,top+30), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,0),2)

def process_detection(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0]*frameWidth)
                center_y = int(detection[1]*frameHeight)
                width = int(detection[2]*frameWidth)
                height = int(detection[3]*frameHeight)
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left,top,width,height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        x = int(left+width/2)
        y = int(top+ height/2)
        drawPredicted(classIds[i], confidences[i], left, top, left+width, top+height,frame,x,y)

