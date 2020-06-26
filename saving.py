import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_pics=0
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("D:\Python\coding blocks data science material\harasscode\haarcascade_frontalface_default.xml")
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

    # if (key_pressed == ord('q')):
    #     break


cap.release()
cv2.destroyAllWindows()

x = face_data[5].reshape((1,100,100,3))
datagen = ImageDataGenerator(rotation_range=40,shear_range=0.2,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2,horizontal_flip=False,fill_mode="nearest")

itr=0
for batch in datagen.flow(x,batch_size=1):
    # plt.figure()
    # plt.imshow(image.img_to_array(batch[0])/255)
    face_data.append(batch[0])
    # plt.axis('off')
    # plt.show()
    itr+=1
    if itr==25:
        break


face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
np.save("D:/Python/test/yolo_opencv3/face_detect_2/data/" + file_name + ".npy", face_data)
