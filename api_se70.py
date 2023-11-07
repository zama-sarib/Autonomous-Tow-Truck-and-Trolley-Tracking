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
