import json
import requests
import datetime

res = {'device_id':"1",'command': "Non Stop",'boolean':False,'startTime': str(datetime.datetime.now())}

requests.post("http://127.0.0.1:5000/", json=res)
# requests.post("http://127.0.0.1:5000/", data="Sarib")