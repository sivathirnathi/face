import cv2 as cv
import os
import numpy as np
import cv2.face
people = []
for i in os.listdir(r"C:\Users\thirn\OneDrive\Desktop\face"):
    people.append(i)
DIR = r"C:\Users\thirn\OneDrive\Desktop\face"
haar_cascade = cv.CascadeClassifier('harcasscade.xml')
features = []
labels = []
def create_train():
    for person in people:
        sub_folder = os.path.join(DIR, person)
        label = people.index(person)
        for img in os.listdir(sub_folder):
            img_path = os.path.join(sub_folder,img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
features = np.array(features, dtype='object')
labels = np.array(labels)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
print("Training completed")
