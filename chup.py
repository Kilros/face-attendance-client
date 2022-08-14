from json.encoder import JSONEncoder
import requests
import json
import cv2,os
import numpy as np
from urllib.request import urlopen
from requests.api import get
from urllib.request import urlopen
from PIL import Image
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    
    # return the image
    return image

def read(url, id, name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        img=url_to_image(url)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imwrite("dataSet/User."+id +'.'+ name + ".jpg", gray[y:y+h,x:x+w])

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
        cv2.imshow("traning",faceNp)
        cv2.waitKey(10)
    return IDs, faces
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path='dataSet'
    Ids,faces=getImagesAndLabels(path)
    #trainning
    recognizer.train(faces,np.array(Ids))
    recognizer.save('recognizer/trainningData.yml')

def cap():
    # cam = cv2.VideoCapture('nhan.mp4')
    cam = cv2.VideoCapture(0)
    detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    id=input('enter your id: ')
    name=input('enter your name: ')
    sampleNum=0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            
            #incrementing sample number 
            sampleNum=sampleNum+1
            #saving the captured face in the dataset folder
            cv2.imwrite("dataSet/User."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame',img)   
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 20
        elif sampleNum>50:
            break
    cam.release()
    cv2.destroyAllWindows()
# id=input("ID: ")
# name=input("Name: ")
# url=input("URL:  ")
# read(url, id, name)

# cap()
# train()
# from datetime import date
# today = date.today()
# print(today);
def upload():
    import requests
    # test_file = open("a.jpeg", "rb")
    test_url = "http://localhost/upload/"
    submit = {'submit': ''}
    files1=[]
    files3=[]
    i=1
    # while True:
    #     files1.append(('file'+str(i), open('dataSet/User.4.'+str(i)+'.jpg', 'rb')))
    #     print(i)
    #     i=i+1
    #     if i==21:
    #         break
    # response1 = requests.post(test_url, data = submit, files = files1)
    # if response1.ok:
    #     print("Upload completed successfully!")
    # else:
    #     print("Something went wrong!")
    while True:
        files=[('file', open('dataSet/User.4.'+str(i)+'.jpg', 'rb'))]
        response= requests.post(test_url, data = submit, files = files)
        if response.ok:
            print(response.text)
        else:
            print("Something went wrong!")
        i=i+1
        if i==51:
            i=1
            break
    
def upload_img():
    URL = "http://localhost/dacn/Ajax/"
    id=8
    token="123"
    sampleNum=1
    submit = {
        'upload_img': '',
        'token': token,
        'id':id
    }
    while True:
        files=[('file', open('dataSet/User.'+str(id)+'.'+str(sampleNum)+'.jpg', 'rb'))]
        response= requests.post(URL+"Upload_img", data = submit, files = files)
        if response.ok:
            print(sampleNum)
            sampleNum= sampleNum+1
        else:
            print("that bai")
        if sampleNum>50:
            break
# upload_img()