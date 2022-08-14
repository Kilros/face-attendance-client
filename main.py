from json.decoder import JSONDecoder
import re
from flask import Flask, render_template, Response,request
import cv2,os
import numpy as np
import requests
import json
from PIL import Image
from datetime import date, datetime
from urllib.request import urlopen

# from chup import upload
app = Flask(__name__)
video = cv2.VideoCapture(0)
work_mode="check_in"
id_get=0
id_temp=0
id_cap=""
name=""
time=""
camera=True
modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# video = cv2.imread("img/a.jpeg")
video = cv2.VideoCapture('Video.mp4')
token="123"
URL = "http://localhost/dacn/Ajax/"
import  os
@app.route('/')
def index():
    # rendering webpage
    today = date.today()
    return render_template('index.html', date=today)
def Getuser(id):
    data = {'getuser': '',
            'id':id,
            'token': token
        }
    test_response = requests.post(URL+"Getuser", data = data)
    if test_response.ok:
        return json.loads(test_response.text)
    else:
        return "false"
def Insert_calendar(id):
    time=datetime.now()
    data = {'insertcalender': '',
            'id':id,
            'in_time':time,
            'token': token
        }
    test_response = requests.post(URL+"Insertcalender", data = data)
    if test_response.ok:
        return json.loads(test_response.text)
    else:
        return False
def Check_out(id):
    out_time=datetime.now()
    data = {'checkout': '',
            'id':id,
            'out_time': out_time,
            'token': token
        }
    response = requests.post(URL+"Check_out_calendar", data = data)
    if response.ok:
        return json.loads(response.text)
    else:
        return False
def gen(video):
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    rec=cv2.face.LBPHFaceRecognizer_create();
    rec.read("recognizer\\trainningData.yml")
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (203,23,252)
    global work_mode
    global id_temp
    global id_get
    global camera
    profile=[]
    # global name
    # global time
    id=0
    while True:
        while camera:
            ret, image = video.read()
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # frame = imutils.resize(frame, width=400)

            # grab the frame dimensions and convert it to a blob
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))


            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                if confidence < 0.5:
                    continue

                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if startX<=w and startY<=h:
            # success, image = video.read()
            # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # faces=faceDetect.detectMultiScale(gray,1.3,5);
            # for(x,y,w,h) in faces:

                    cv2.rectangle(image,(startX,startY),(endX,endY),(255,0,0),2)
                    id,conf=rec.predict(gray[startY:endY,startX:endX])
                    print(conf)
                    
                    if (conf<45):
                        if(id!=id_get):
                            profile=Getuser(id)
                            print("get")
                        if(profile!=[]):
                            # if(id_temp==profile[0]["id"]): 
                            #     cv2.putText(image, ""+ profile[0]["fullname"], (x,y+h+30), fontface, fontscale, fontcolor ,2)
                            #     cv2.putText(image, "DA DIEM DANH ROI", (x,y+h+60), fontface, fontscale, fontcolor ,2)  
                            #     break
                            # print(work_mode)
                            if(id!=id_get):
                                if work_mode=="check_in":
                                    insert=Insert_calendar(profile[0]["id"]) 
                                    # print(insert)
                                else:
                                    insert=Check_out(profile[0]["id"]) 
                                    print(insert)
                            if(insert=="exist"): 
                                id_get=id
                                # cv2.putText(image, ""+ profile[0]["fullname"], (startX,endY+30), fontface, fontscale, fontcolor ,2)  
                                cv2.putText(image, "DA DIEM DANH", (startX,endY+30), fontface, fontscale, fontcolor ,2)     
                                break          
                            if(insert):                      
                                id_temp=profile[0]["id"]  
                                # name=profile[0]["fullname"]
                                # time=datetime.now()
                                id_get=id
                                insert="exist"
                                print('thêm thành công')
                                cv2.putText(image, ""+ profile[0]["fullname"], (startX,endY+30), fontface, fontscale, fontcolor ,2)                    
                        else:
                            cv2.putText(image, "Unknown", (startX,endY+30), fontface, fontscale, fontcolor ,2)
                            id_get=0     
                            id_temp=0
                    else:
                        cv2.putText(image, "Unknown", (startX,endY+30), fontface, fontscale, fontcolor ,2) 
                        id_get=0   
                        id_temp=0   
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video')
def video_feed():
    global video
    return Response(gen(video),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/Getcalendar',methods=['POST'])
def getcalendar():
    # global name
    # global time
    # return json.dumps(name)
    global id_get
    data = {'getcalendar': '',
            'token': '123',
            'id': id_get
        }
    try:
        test_response = requests.post(URL+"Getcalender", data = data)
        if test_response.ok:
            return test_response.text
    except:
        print("err post")
@app.route('/stop',methods=['POST'])
def stop():
    global camera
    if request.method == 'POST':
        action = request.form['action']
        # print(ac)
        if action=="stop":
            camera=False
            return "stop"
        if action=="start":
            camera=True
            return "start" 
@app.route('/mode',methods=['POST'])
def mode():
    global work_mode
    global id_get
    if request.method == 'POST':
        mode_change = request.form['mode']
        try:
            work_mode=mode_change
            id_get=0
            return "True"
        except:
            return "false"
@app.route('/Capture')
def capture():
    # rendering webpage
    today = date.today()
    return render_template('capture.html', date=today)
def cap(video):
    detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    global id_cap
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (203,23,252)
    sampleNum=0
    while True:
        ret, image = video.read()
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))


        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startX<=w and startY<=h:
        # success, image = video.read()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # faces = detector.detectMultiScale(gray, 1.3, 5)
        # for (x,y,w,h) in faces:
                cv2.rectangle(image,(startX,startY),(endX,endY),(255,0,0),2)
                if (id_cap!=""):
                    sampleNum=sampleNum+1
                    cv2.imwrite("dataSet/User."+id_cap +'.'+ str(sampleNum) + ".jpg", gray[startY:endY,startX:endX])
                    if upload_img(id_cap, sampleNum):
                        print("Upload thành công")
                    else:
                        print("Upload thất bại")
                    if sampleNum>=50:
                        sampleNum=0
                        id_cap=""
                        cv2.putText(image, "CHUP THANH CONG", (startX,endY+60), fontface, fontscale, fontcolor ,2)                  
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def upload_img(id, sampleNum):
    submit = {
        'upload_img': '',
        'token': token,
        'id':id
    }
    files=[('file', open('dataSet/User.'+str(id)+'.'+str(sampleNum)+'.jpg', 'rb'))]
    response= requests.post(URL+"Upload_img", data = submit, files = files)
    if response.ok:
        return True
    else:
        return False
@app.route('/Capture_video')
def cap_video():
    global video
    return Response(cap(video),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/Getuser_id',methods=['POST'])
def getuser():
    if request.method == 'POST':
        id = request.form['id']
        return json.dumps(Getuser(id))
@app.route('/cap',methods=['POST']) 
def cap_id():
    global id_cap
    if request.method == 'POST':
        id = request.form['id']
        if id!="":
            id_cap=id
            print(id_cap)
            return "true"
        else:
            return "false"
@app.route('/train',methods=['POST']) 
def recognizer():
    for item in Getimg():
        basename = os.path.basename(item["thumbnail"])
        img=url_to_image("http://localhost/dacn/"+item["thumbnail"])
        cv2.imwrite("dataSet/"+basename, img)
    train()
    return "True"
def Getimg():
    data = {'getimg': '',
            'token': token
        }
    test_response = requests.post(URL+"Getimg", data = data)
    if test_response.ok:
        return json.loads(test_response.text)
    else:
        return False
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    # return the image
    return image
def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        #split to get ID of the image
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
    return IDs, faces
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path='dataSet'
    Ids,faces=getImagesAndLabels(path)
    #trainning
    recognizer.train(faces,np.array(Ids))
    recognizer.save('recognizer/trainningData.yml')
@app.route('/getimg_id',methods=['POST']) 
def getimg_id():
    if request.method == 'POST':
        id = request.form['id']
        return json.dumps(Getimg())
if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="localhost", port=8000, debug=True)
