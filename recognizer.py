import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('harcasscade.xml')
people = []
for i in os.listdir(r"C:\Users\thirn\OneDrive\Desktop\face"):
    people.append(i)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)
        cv.putText(frame, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.putText(frame, str(confidence), (200,200), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
