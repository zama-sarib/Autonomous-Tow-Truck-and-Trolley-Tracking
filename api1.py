from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
import json
from datetime import datetime

app = Flask(__name__)
api = Api(app)


@app.route('/',methods = ['POST', 'GET'])
def get():
    data = request.get_json()
    print(datetime.now()-datetime.strptime(data['startTime'], '%Y-%m-%d %H:%M:%S.%f'))
    return jsonify("Action taken !!")

if __name__ == "__main__":
    app.run(debug=True)