#!/usr/bin/env python

import rospy
from flask import Flask, request
# from dbw.srv import *
app = Flask(__name__)


@app.route('/status_depth_d435i', methods=['POST'])
def send_data():
    received_data = request.form.get('data')
    print("Received data:", received_data)
    if(received_data == "STOP"):
        print(received_data)
        # emergency_stop(1,0,"d435_depth")
    else:
        print(received_data)
        # emergency_stop(0,0,"d435_depth")
    return 'Data received successfully'

if __name__== "__main__":
    rospy.init_node( 'data_reciver_gpu', anonymous=True)
    # emergency_stop = rospy.ServiceProxy('emergencyStop',emergencyStop)
    app.run(host='0.0.0.0', port=5001)  


       
    
    
    
    rospy.spin()
