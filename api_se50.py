from flask import Flask, render_template, Response
import cv2
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/status_depth_d435i',methods=['POST'])
def status_depth_d435i():
    jsondata = request.form['jsondata']
    print(f'Status: {jsondata}')
    result = {'result':'Received'}
    return json.dumps(result)

    


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
