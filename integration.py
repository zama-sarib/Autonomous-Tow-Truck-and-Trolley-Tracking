
import numpy as np
import cv2
import torch
from super_gradients.training import models
import scipy


dataset_params = {
    'classes': ['Trolley','Person']
}

best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="/content/checkpoints/my_first_yolonas_run/ckpt_latest.pth")
    
    
cap = cv2.VideoCapture(0)      
width,height = 416,416



def getTrolleyCoords(images_predictions):
    '''
    This function returns the trolley coords detected in the frame.
    '''
    
    coords = []
    for image_prediction in images_predictions:
        
        labels = image_prediction.prediction.labels
        confidence = image_prediction.prediction.confidence
        bboxes = image_prediction.prediction.bboxes_xyxy
        
        if labels[0] == 0 and confidence[0] >= 0.8: # 0 --> Trolley class
            mid_x = ((bboxes[0][1]+bboxes[0][3])/2)
            mid_y = ((bboxes[0][0]+bboxes[0][2])/2)
            coords.append([mid_x,mid_y])
        
        coords = np.array(coords)
    
    return coords


def getPersonCoords(images_predictions):
    '''
    This function returns the Person coords detected in the frame.
    '''
    
    coords = []
    for image_prediction in images_predictions:
        
        labels = image_prediction.prediction.labels
        confidence = image_prediction.prediction.confidence
        bboxes = image_prediction.prediction.bboxes_xyxy
        
        if labels[0] == 1 and confidence[0] >= 0.8: # 0 --> Trolley class
            mid_x = ((bboxes[0][1]+bboxes[0][3])/2)
            mid_y = ((bboxes[0][0]+bboxes[0][2])/2)
            coords.append([mid_x,mid_y])
        
        coords = np.array(coords)
    
    return coords
    
    
    
        
while True:
    success,frame = cap.read()
    img = cv2.resize(frame,(width, height))
    
    if success:    
        images_predictions = best_model.predict(img,iou=0.7, conf=0.8)
        
        trolley_coord = getTrolleyCoords(images_predictions)
        person_coord = getPersonCoords(images_predictions)
                
        distance_array = scipy.spatial.distance.cdist(trolley_coord,person_coord)[0]
        indices = np.where(distance_array <= 0.5)[0]
        
        if indices.shape[0] > 1:
            # indices = indices[0]
            # if distance_array[indices] <= 0.5: # Threhold is set to 0.5
            
            pass # #Threshold let say is 0.5 unit distance trigger the stop module.
                
                    
                
                
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

        
pipeline.stop()
cv2.destroyAllWindows()
