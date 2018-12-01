#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import sys
import os, time
import easygui
from libFacialDoor import webCam
from libFacialDoor import facenetVerify

GRAPH_FILENAME = "facenet_celeb_ncs.graph"
FACE_MATCH_THRESHOLD = 0.75
webcam_size = (320,240)

previewPicPath = "preview/"
validPicPath = "valid/"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_scale = 1.1
cascade_neighbors = 8
minFaceSize = (90,100)

#cv2.namedWindow("SunplusIT", cv2.WND_PROP_FULLSCREEN)        # Create a named window
#cv2.setWindowProperty("SunplusIT", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#-----------------------------------------------------------------------

def createEnv():
    if not os.path.exists(validPicPath):
        os.makedirs(validPicPath)
        print("Pics for valid path created:", validPicPath)

    if not os.path.exists(previewPicPath):
        os.makedirs(previewPicPath)
        print("Pics for preview path created:", previewPicPath)

def getFaces_cascade(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= cascade_scale,
        minNeighbors=cascade_neighbors,
        minSize=minFaceSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    bboxes = []
    for (x,y,w,h) in faces:
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            bboxes.append((x, y, w, h))

    return bboxes

def chkID(id):
    if not os.path.exists(validPicPath + str(id)):
        easygui.msgbox('這個ID{}還沒有申請刷臉打卡！'.format(id))
        return False
    else:
        if(os.path.exists(validPicPath + str(id) + '/cam0/valid.jpg') and \
                (os.path.exists(validPicPath + str(id) + '/cam1/valid.jpg'))):
            return True
        else:
            return False

#------------------------------------------------------------------------

camOK = True
cam1 = webCam(id=0, size=(320,240))
if(cam1.working() is False):
    camOK = False
    print("Web camera #1 is not working!")

cam2 = webCam(id=1, size=(320,240))
if(cam2.working() is False):
    camOK = False
    print("Web camera #2 is not working!")

faceCheck = facenetVerify(graphPath=GRAPH_FILENAME, movidiusID=0)
#------------------------------------------------------------------------

while camOK:

    okPic1 = True
    okPic2 = True

    okPic1, pic1 = cam1.takepic(rotate=0, resize=None, savePath=None)
    if(okPic1 is not True):
        print("Taking a picture by cam1 is failed!")

    okPic2, pic2 = cam2.takepic(rotate=0, resize=None, savePath=None)
    if(okPic2 is not True):
        print("Taking a picture by cam2 is failed!")

    if(okPic1 is True):
        valid = cv2.imread(validPicPath + "200334/cam0/valid.jpg")
        faceCheck.face_match(face1=pic1, face2=valid, threshold=0.75)

    if(okPic2 is True):
        valid = cv2.imread(validPicPath + "200334/cam0/valid.jpg")
        faceCheck.face_match(face1=pic2, face2=valid, threshold=0.75)

    cv2.imshow("cam1", pic1)
    cv2.imshow("cam2", pic2)

    cv2.waitKey(1)
