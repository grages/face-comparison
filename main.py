import getphoto
import detect
#import align
import extract
import classify
import show
import numpy as np
import cv2
import copy
import time

tag1=0
tag2=0
tag3=0
cosof=0.0
cam=getphoto.initcamera()
cascade=detect.initvj()
show.initwindow()
vgg,transformer=extract.initvgg()
cla=classify.initgnb()

while (1):
    cv2.waitKey(10)
    image=getphoto.getphoto(cam)
    show.show(image,0)
    time1=time.time()
    [x1,y1,x2,y2,tag]=detect.face(image,cascade)
    time2=time.time()-time1
    print('Face Detect cost: %f' %time2)
    if tag==1:
        image=detect.split(image,x1,y1,x2,y2)
        #tag=0
        #[image,tag]=align.face(image,cascade)
    if tag==1:
        show.show(image,3)
        key=cv2.waitKey(800) & 0xFF
        if key==49:
            tag1=1
            ima1=image.copy()
            show.show(ima1,1)
            tag3=0
        if key==50:
            tag2=1
            ima2=image.copy()
            show.show(ima2,2)
            tag3=0
        if key==27:
            break
    if (tag1==1 and tag2==1 and tag3==0):
        time1 = time.time()
        ex1=copy.deepcopy(extract.face(ima1,vgg,transformer))
        time2 = time.time() - time1
        print('Extract1 cost: %f' % time2)

        time1 = time.time()
        ex2=copy.deepcopy(extract.face(ima2,vgg,transformer))
        time2 = time.time() - time1
        print('Extract2 cost: %f' % time2)
        cosof=classify.calcos(ex1,ex2)
        print('cosof %f'%cosof)
        result=classify.gnb(cosof,cla)
        tag3=1
        print(result)
        show.showresult()

getphoto.release(cam)
show.release()