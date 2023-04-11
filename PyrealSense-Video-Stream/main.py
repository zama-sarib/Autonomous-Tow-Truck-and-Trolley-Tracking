from flask import Flask, render_template, Response
import cv2
import pyrealsense as pyreal
import numpy as np
import pyrealsense2 as rs
import sys

app = Flask(__name__)

def gen():
    
    video = cv2.VideoCapture(0)
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    while True:
    # Wait for a coherent pair of frames: depth and color
    # frames = pyreal.pipeline.wait_for_frames()
    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()
    # if not depth_frame or not color_frame:
    #     continue
    # Convert images to numpy arrays
    # depth_image = np.asanyarray(depth_frame.get_data())
    # color_image = np.asanyarray(color_frame.get_data())
        success,color_image = video.read()
        if success:
            blob = cv2.dnn.blobFromImage(color_image, 1/255, (pyreal.inpWidth, pyreal.inpHeight), [0,0,0],1,crop=False)
            net.setInput(blob)
            outs = net.forward(pyreal.getOutputsNames(net))
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            pyreal.process_detection(color_image,outs)
            # depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            # If depth and color resolutions are different, resize color image to match depth image for display
            # if depth_colormap_dim != color_colormap_dim:
            #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
            #                                      interpolation=cv2.INTER_AREA)
            #     images = np.hstack((resized_color_image, depth_colormap))
            # else:
            #     images = np.hstack((color_image, depth_colormap))
            # Show images
            # cv2.imshow('Yolo in RealSense made by Tony', images)
            ret, jpeg = cv2.imencode('.jpg', color_image)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(jpeg) + b'\r\n\r\n')
        else:
            break
        # Stop streaming
        # pyreal.pipeline.stop()
        


@app.route('/')
def index():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,threaded=True)