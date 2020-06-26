import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from sklearn.linear_model import LogisticRegression

num_pics = 0
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    "D:\Python\coding blocks data science material\harasscode\haarcascade_frontalface_default.xml")
skip = 0
face_data = []
file_name = input("Enter your name:")
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    if len(faces) >= 1:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (85, 27, 25), 2)
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        skip += 1
        if skip % 5 == 0:
            print(len(face_data))
            num_pics += 1
            face_data.append(face_section)
            if num_pics == 18:
                break

    cv2.imshow("Say Cheese....", frame)

    key_pressed = cv2.waitKey(1) & 0xFF

    # if (key_pressed == ord('#')):
    #     break

cap.release()
cv2.destroyAllWindows()

x = face_data[5].reshape((1, 100, 100, 3))
datagen = ImageDataGenerator(rotation_range=40, shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
                             zoom_range=0.2, horizontal_flip=False, fill_mode="nearest")

itr = 0
for batch in datagen.flow(x, batch_size=1):
    # plt.figure()
    # plt.imshow(image.img_to_array(batch[0])/255)
    face_data.append(batch[0])
    # plt.axis('off')
    # plt.show()
    itr += 1
    if itr == 32:
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
np.save("D:/Python/test/yolo_opencv3/face_detect_2/data/" + file_name + ".npy", face_data)


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