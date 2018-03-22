import numpy as np
import cv2

def initwindow():
    a=1
    return a

def show(image,num):
    if num==0:
        cv2.imshow('Camera',image)

    elif num==1:
        cv2.imshow('Image_1',image)

    elif num==2:
        cv2.imshow('Image_2',image)

    elif num==3:
        cv2.imshow('Detected',image)


def showresult():
    a=1
    return a

def release():
    cv2.destoryAllWindows()
    return 0
