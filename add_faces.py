import cv2
import pickle
import os
import numpy as np
face_detect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
facedata = []
i=0
name = input("Enter name of the student: ")
#display video
while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(grey, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h,x:x+w,:]
        resized_img = cv2.resize(crop_img, (50,50))
        if len(facedata)<=100 and i%10==0:
            facedata.append(resized_img)
        i+=1
        cv2.putText(frame, str(len(facedata)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,250), 2)
    cv2.imshow('image', frame)
    if cv2.waitKey(1)==ord('x') or len(facedata)==100:
        break

cap.release()
cv2.destroyAllWindows()

choice = int(input("Enter choice: "))

if choice==1:
    facedata = np.asarray(facedata)
    facedata = facedata.reshape(100,-1)


    if 'names.pkl' not in os.listdir('data/'):
        names = [name]*100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names,f)

    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names += [name]*100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(facedata,f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        face_img = pickle.load(f)
    face_img = np.append(face_img,facedata, axis=0)
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(face_img, f)
        