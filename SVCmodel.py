from sklearn.svm import SVC
import cv2
import pickle
import numpy as np
face_detect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
facedata = []
i=0
with open('data/names.pkl','rb') as f:
    LABELS = pickle.load(f)
with open('data/face_data.pkl','rb') as f:
    FACES = pickle.load(f)

#print(LABELS)
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(FACES,LABELS)
facedata_x=[]
i=0

#display video
while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(grey, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h,x:x+w,:]
        resized_img = cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        if len(facedata_x)<=20:
            facedata_x.append(resized_img)
        i+=1
        cv2.putText(frame, 'scanning...', (x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
        cv2.putText(frame, str(len(facedata_x)*5)+'%', (x+100,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,250), 2)
    cv2.imshow('image', frame)
    if cv2.waitKey(1)==ord('x') or len(facedata_x)==20:
        break

cap.release()
cv2.destroyAllWindows()

facedata_x = np.asarray(facedata_x)
facedata_x = facedata_x.reshape(20,-1)
pred = model.predict(facedata_x)

if len(np.unique(pred))==1:
    print('Prediction: ', pred[0])
    print(pred[0],'is present.')
else:
    print("Cannot find student !!! Try again.")

