import numpy as np
import cv2


def initvj():
    face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_alt2.xml")
    return face_cascade

def face(image,face_cascade):
    ima=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(ima, 1.1, 2)
    result = []
    if len(faces)>=1:
        for (x, y, width, height) in faces:
            result.extend([x, y, x + width, y + height])
            break
        result.append(1)
        return result
    else:
        return [1,1,1,1,0]

def split(image,x1,y1,x2,y2):
    ima=image[y1:y2,x1:x2]
    return ima
