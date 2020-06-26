import numpy as np
import cv2
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dataset_path = "D:/Python/test/yolo_opencv3/face_detect_2/data/"
face_data = []
label = []
class_id = 0
names = {}
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_data = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(label, axis=0).flatten()

model = LogisticRegression()
model_fit = model.fit(face_data,face_labels)
cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("D:\Python\coding blocks data science material\harasscode\haarcascade_frontalface_default.xml")

skip=0
while True:
    ret,frame = cap.read()
    if(ret==False):

        continue
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
    for (x,y,w,h) in faces:
        pred_name=""
        offset=10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]

        if face_section.shape[0]>=100 and face_section.shape[1]>=100:
            face_section = cv2.resize(face_section,(100,100))
            out =  model.score(face_section.reshape((1,-1)),[0])
            if out==1:
                pred_name = names[0]
            else:
                pred_name="Name Not Found"
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Hello ",frame)
    key = cv2.waitKey(1)& 0xFF
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()