
import numpy as np
import cv2
import torch
from super_gradients.training import models
import scipy
import requests



inputDict = {
    "Vehicle_ID" : "",
    "Command" : ""        
}

dataset_params = {
    'classes': ['Trolley','Person']
}


best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="/content/checkpoints/my_first_yolonas_run/ckpt_latest.pth")
    
    
cap = cv2.VideoCapture(0)      
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
    while True:
        success,frame = cap.read()
        img = cv2.resize(frame,(width, height))
        
        if success:    
            images_predictions = best_model.to(device).predict(img,iou=0.7, conf=0.7)
            trolley_coord,person_coord = getCoords(images_predictions)
            
            for person in person_coord:
                leftResponse = isPersonLeft(person,trolley_coord)
                rightResponse = isPersonRight(person,trolley_coord)
                
                possibleDanger = False
                MinDistance = 1e9

                if leftResponse == True and rightResponse == True:
                    possibleDanger = False
                elif leftResponse == True and rightResponse == False:
                    possibleDanger = True
                else:
                    possibleDanger = True
                
                if possibleDanger:
                    if leftResponse:
                        MinDistance = getLeftMinDistance(person,trolley_coord)
                    else:
                        MinDistance = getRightMinDistance(person,trolley_coord)

                                
        
            
            if MinDistance < 100:             
                #Threshold let say is 200 unit distance triggers the stop module.
                endpoint = f'https://127.0.0.1:1003/emergencyStop'
                response = requests.post(endpoint,params=inputDict)


        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

    cv2.destroyAllWindows()
    
    
    

if __name__ == '__main__':
    start_processing()
    
    
# lan hub