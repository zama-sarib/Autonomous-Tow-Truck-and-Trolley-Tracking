import cv2
import numpy as np
import pyrealsense2 as rs
import torch

# Load model with pretrained weights
from super_gradients.training import models
from super_gradients.common.object_names import Models


model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
device = 0 if torch.cuda.is_available() else "cpu"

# Create a context object. This object owns the handles to all connected RealSense devices
pipeline = rs.pipeline()
pipeline.start()

# Define the coordinates of the top-left and bottom-right corners of your ROI
roi_top_left = (200, 150)  # (x, y) coordinates
roi_bottom_right = (400, 350)

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

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.03), cv2.COLORMAP_JET)

    output = model.to(device).predict(color_data,iou=0.7, conf=0.7)

    # Create a mask for the ROI
    mask = np.zeros_like(depth_data)
    mask[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = 1

    # Apply the mask to the depth data to get the depth values within your ROI
    depth_roi = np.multiply(depth_data, mask)

    # Check if any depth value in the ROI is less than the threshold in meters
    if np.any(depth_roi < threshold_depth_meters):
        print("Depth in ROI is less than the threshold of {} meters!".format(threshold_depth_meters))
    else:
        print("Depth in ROI is greater than or equal to the threshold of {} meters.".format(threshold_depth_meters))

    # Draw a rectangle around the ROI on the RGB image
    #cv2.rectangle(color_data, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

    # Display the RGB image with the marked ROI
    # cv2.imshow("RGB Image with ROI", color_data)
    for i,predicted_frame in enumerate(output):
        frame = predicted_frame.draw()
        #cnt = np.count_nonzero(predicted_frame.prediction.labels == 0.0)
        #cv2.putText(frame,str(cnt),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,225),3)
        depth_colormap_dim = depth_data.shape
        color_colormap_dim = frame.shape
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(frame, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                interpolation=cv2.INTER_AREA)
            print("After changes",resized_color_image.shape,depth_data.shape)
#            images = np.hstack((resized_color_image, depth_data))
        else:
            images = np.hstack((frame, depth_data))
        cv2.imshow("frame",frame)

    # Checking for user input 
    # key = cv2.waitKey(1)
    # if key == 27:  # Exit on ESC key value is 27
    #     break
    if cv2.waitKey(1000) & 0xFF == ord('q'):
            break 


# Release resources and close OpenCV windows
cv2.destroyAllWindows()

