
import numpy as np
import cv2
import time

def initcamera(num=0):
    cap = cv2.VideoCapture(num)
    print("Camera opening...")
    while (not cap.isOpened()):
        cap.open()
        key=cv2.waitkey(10) & 0xFF
        if key==27:
            print("Camera initialize stoped.")
    print("Camera opened.")
    return cap

def getphoto(cap):
    ret,frame=cap.read()
    tmp=0
    while (not ret):
        ret, frame = cap.read()
        tmp+=1
        cv2.waitkey(10)
        if (tmp>100):
            print('Can\'t get photo')
            return
    return frame

def release(cap):
    cap.release()