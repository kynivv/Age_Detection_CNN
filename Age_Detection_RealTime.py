import cv2
import imutils
import numpy as np
from keras.models import load_model
import warnings

warnings.filterwarnings('ignore')

model = load_model('final_cnn_model_checkpoint.h5')

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']

cam = cv2.VideoCapture(0)

while 1:
    ret, frame = cam.read()
    if ret:
        faces = detector.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (200, 200))
            face = face.reshape(1, 200, 200, 1)
            age = model.predict(face)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color=(140, 0, 190), thickness=2)
            cv2.rectangle(frame,(x, y+h), (x+w, y+h+50), color=(140,0,190), thickness=-1)
            cv2.putText(frame, ranges[np.argmax(age)], (x+65, y+h+35),cv2.FONT_HERSHEY_DUPLEX, 0.8,(255,255,255),2)

        cv2.imshow('Live', frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()