
import numpy as np
import cv2
import torch
from super_gradients.training import models
import scipy
import requests
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



inputDict = {
    "Vehicle_ID" : "",
    "Command" : ""        
}

dataset_params = {
    'classes': ['Trolley','Person']
}


best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path=r"C:\Users\samri\Desktop\Realsense CV\trolley\ckpt_best.pth")
    
    
cap = cv2.VideoCapture(r"C:\Users\samri\Downloads\RealSense Video Feed 11 July.mp4")      
width,height = 416,416


def getCoords(images_predictions):
    '''
    This function returns the midpoint Person and Trolley coords detected in the frame.
    '''

    Person_coords = []
    trolley_coords = []
    for i,image_prediction in enumerate(images_predictions):

        labels = image_prediction.prediction.labels
        confidence = image_prediction.prediction.confidence
        bboxes = image_prediction.prediction.bboxes_xyxy

    for i in range(len(labels)):
        if labels[i] == 0.0: # 0 --> Trolley class
            trolley_coords.append(bboxes[i])

        elif labels[i] == 1.0: # 0 --> Person class
            Person_coords.append(bboxes[i])

    trolley_coords = np.array(trolley_coords)
    Person_coords = np.array(Person_coords)

    return trolley_coords,Person_coords



def isPersonLeft(person,trolley_coords):
    '''
    This Function evaluates whether the person is on the left of the trolley
    '''
    
    x2_person = person[2]
    boolean = True
    for eachCoord in trolley_coords:
        if x2_person < eachCoord[2]:
            pass
        else:
            boolean = False
    return boolean



def isPersonRight(person,trolley_coords):
    '''
    This Function evaluates whether the person is on the right of the trolley
    '''
    
    x1_person = person[0]
    boolean = True
    for eachCoord in trolley_coords:
        if x1_person > eachCoord[0]:
            pass
        else:
            boolean = False
    return boolean



def getLeftMinDistance(person,trolley_coords):
    '''
    This Function evaluates the minimum distance between the person and the nearest right trolley
    '''
    
    minDistance = 1e9
    for eachCoord in trolley_coords:
        minDistance = min(minDistance,eachCoord[0]-person[2])
    
    return minDistance



def getRightMinDistance(person,trolley_coords):
    '''
    This Function evaluates the minimum distance between the person and the nearest left trolley
    '''
    
    minDistance = 1e9
    for eachCoord in trolley_coords:
        minDistance = min(minDistance,person[0]-eachCoord[2])
    
    return minDistance



def start_processing():    
    '''
    This function start the video capture and passes it to DL model.
    '''
    device = 0 if torch.cuda.is_available() else "cpu"
    # print(device)
    # while cap.isOpened():
    while True:
        success,frame = cap.read()
        img = cv2.resize(frame,(width, height))
        
        if success:    
            images_predictions = best_model.to(device).predict(img,iou=0.7, conf=0.7)
            # images_predictions.show()

            for i,predicted_frame in enumerate(images_predictions):
                frame = predicted_frame.draw()
                cv2.imshow("frame",frame)
            trolley_coord,person_coord = getCoords(images_predictions)
            
            for person in person_coord:
                leftResponse = isPersonLeft(person,trolley_coord)
                rightResponse = isPersonRight(person,trolley_coord)
                
                possibleDanger = False
                MinDistance = 1e9

                if leftResponse == True and rightResponse == True:
                    possibleDanger = False
                    print("Middle")
                elif leftResponse == True and rightResponse == False:
                    possibleDanger = True
                    print("Left")
                else:
                    possibleDanger = True
                    print("Right")
                
                if possibleDanger:
                    if leftResponse:
                        MinDistance = getLeftMinDistance(person,trolley_coord)
                    else:
                        MinDistance = getRightMinDistance(person,trolley_coord)

                                
                print(MinDistance)
            
            # if MinDistance < 100:             
                #Threshold let say is 200 unit distance triggers the stop module.
                # endpoint = f'https://127.0.0.1:1003/emergencyStop'
                # response = requests.post(endpoint,params=inputDict)

            # cv2.imshow('video',img)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break 
 
    cap.release()

cv2.destroyAllWindows()
    
    
    

if __name__ == '__main__':
    # start_processing()
    device = 0 if torch.cuda.is_available() else "cpu"
    test_img = r"C:\Users\samri\Desktop\Realsense CV\trolley\DataSet\Train\Images\64.jpg"
    img = cv2.imread(test_img)
    # cv2.imshow("image", img)
    # images_predictions =  best_model.predict(img,iou=0.7,conf=0.7)  
    # images_predictions.show()
    # cv2.waitKey(0)
    images_predictions = best_model.to(device).predict(img,iou=0.7, conf=0.7)
            # images_predictions.show()

    
    trolley_coord,person_coord = getCoords(images_predictions)
    
    for person in person_coord:
        leftResponse = isPersonLeft(person,trolley_coord)
        rightResponse = isPersonRight(person,trolley_coord)
        
        possibleDanger = False
        MinDistance = 1e9

        if leftResponse == True and rightResponse == True:
            possibleDanger = False
            print("Middle")
        elif leftResponse == True and rightResponse == False:
            possibleDanger = True
            print("Left")
        else:
            possibleDanger = True
            print("Right")
        
        if possibleDanger:
            if leftResponse:
                MinDistance = getLeftMinDistance(person,trolley_coord)
            else:
                MinDistance = getRightMinDistance(person,trolley_coord)

                        
        print(f"Mindistance: {MinDistance}")
    
            # if MinDistance < 100:             
                #Threshold let say is 200 unit distance triggers the stop module.
                # endpoint = f'https://127.0.0.1:1003/emergencyStop'
                # response = requests.post(endpoint,params=inputDict)

            # cv2.imshow('video',img)
        
    for i,predicted_frame in enumerate(images_predictions):
        frame = predicted_frame.draw()
        cnt = np.count_nonzero(predicted_frame.prediction.labels == 0.0)
        cv2.putText(frame,str(cnt),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,225),3)
        cv2.imshow("frame",frame)
        

    cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    
# lan hub