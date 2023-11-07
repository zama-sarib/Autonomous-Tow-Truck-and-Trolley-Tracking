import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time 
import sys
from ultralytics import YOLO

import requests
import time

receiver_ip = '192.168.0.4'  # IP address of the receiver machine
receiver_port = 5001  # Port on which the receiver machine is listening
api = 'status_depth_d435i'
distance_threshold = 1.5

# Initialize the parameters
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416
classesFile = "coco.names"

model = YOLO('yolov8n.pt')
device1 = 0 if torch.cuda.is_available else 'cpu'
print(torch.cuda.get_device_name())
print(device1)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
found_rgb = False

for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    sys.exit()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)

if __name__ == "__main__":
    start_time = time.time()
# FPS update time in seconds
    display_time = 0
    fc = 0
    FPS = 0
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                print(type(color_image),color_image.shape)
                
            result = model.to(device1)(color_image)
            annotated_frame = result[0].plot()
            for r in result:
                for bboxes in r.boxes.xyxy:
                    x_mid = (bboxes[0] + bboxes[2]) / 2
                    y_mid = (bboxes[1] + bboxes[3]) / 2
                    zDepth = depth_frame.get_distance(int(x_mid),int(y_mid))
                    if zDepth < distance_threshold:
                        stop_cmd = True
                        break
                    cv2.circle(annotated_frame,(int(x_mid),int(y_mid)),2,(255, 0, 0),2)
                    print(f"Distance to Object: {zDepth}")
            
            images = np.hstack((annotated_frame, depth_colormap))
            fc+=2
            TIME = time.time() - start_time
            if(stop_cmd ==True):
                pass
	         # Call the API
		
            if (TIME) >= display_time :
                FPS = fc / (TIME)
                fc = 0
                start_time = time.time()

            fps_disp = "FPS: "+str(FPS)[:5]
            cv2.putText(images, fps_disp, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Yolo in RealSense', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop streaming
        pipeline.stop()