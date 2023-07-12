import cv2
import os 

data_path = os.getcwd()+'/data'
if not os.path.isdir(data_path):
    os.mkdir(data_path)

image_id = 1
parent_dir = os.getcwd()
video_dir = parent_dir+'/videos'
saved_dir = parent_dir+'/data'
for file in os.listdir(video_dir):
    video_path = os.path.join(video_dir , file)
    video = cv2.VideoCapture(video_path)
    success = True
    count = 1

    while success:
        success , frame = video.read() 
        if success == True:
            if count%5 == 0:
                name = str(image_id)+".jpg"
                image_id += 1
                frame = cv2.resize(frame,(416,416))
                cv2.imwrite(os.path.join(saved_dir , name),frame)
            count += 1
        else:
            break
    print("Extracted Frames from {}: {}".format(file,image_id))

print("Total Extracted Frames :",image_id) 

