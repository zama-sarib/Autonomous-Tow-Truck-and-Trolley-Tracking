import cv2
import numpy as np
import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected RealSense devices
pipeline = rs.pipeline()
pipeline.start()

# Define the coordinates of the top-left and bottom-right corners of your ROI
roi_top_left = (200,150)  # (x, y) coordinates
roi_bottom_right = (400, 200)

# Define a threshold value in meters for the depth check
threshold_depth_meters = 1.0  # Adjust this threshold value as needed in meters

while True:
    # This call waits until a new coherent set of frames is available on a device
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_data = np.asanyarray(depth_frame.get_data())

    # Get the RGB frame
    color_frame = frames.get_color_frame()
    color_data = np.asanyarray(color_frame.get_data())

    # Create a mask for the ROI
    mask = np.zeros_like(depth_data)
    mask[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = 1

    # Apply the mask to the depth data to get the depth values within your ROI
    depth_roi = np.multiply(depth_data, mask)
    max_depth = np.min(depth_roi)
    max_depth_coords = np.unravel_index(np.argmax(depth_roi), depth_roi.shape)
    #print(type(max_depth_coords))
    #print(max_depth_coords[0],max_depth_coords[1])
    print(depth_frame.get_distance(max_depth_coords[0],max_depth_coords[1]))
    # Check if any depth value in the ROI is less than the threshold in meters
    #if np.any(depth_roi < threshold_depth_meters):
        #print("Depth in ROI is less than the threshold of {} meters!".format(threshold_depth_meters))
    #else:
        #print("Depth in ROI is greater than or equal to the threshold of {} meters.".format(threshold_depth_meters))

    # Draw a rectangle around the ROI on the RGB image
    cv2.rectangle(color_data, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

    # Display the RGB image with the marked ROI
    cv2.imshow("RGB Image with ROI", color_data)

    # Checking for user input 
    key = cv2.waitKey(1)
    if key == 27:  # Exit on ESC key value is 27
        break

# Release resources and close OpenCV windows
cv2.destroyAllWindows()

import requests
import time

receiver_ip = '192.168.0.4'  # IP address of the receiver machine
receiver_port = 5001  # Port on which the receiver machine is listening
api = 'status_depth_d435i'
while True:
    data_to_send = 'START'
    try:
    	response = requests.post(f'http://{receiver_ip}:{receiver_port}/{api}', data={'data': data_to_send})
    	print(response.text)
    except:
    	print("An error occurred while sending the request")
    
    
    time.sleep(0.1) 
