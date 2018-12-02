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
tmpPic1 = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
tmpPic2 = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
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

def matchFace():
    global tmpPic1, tmpPic2

    camOK = True
    cam1 = webCam(id=0, size=(320,240))
    if(cam1.working() is False):
        camOK = False
        print("Web camera #1 is not working!")

    cam2 = webCam(id=1, size=(320,240))
    if(cam2.working() is False):
        camOK = False
        print("Web camera #2 is not working!")

    if(camOK is not True):
        print("web camera is not working!")
        return None, None, None

    totalCount1 = 0
    totalCount2 = 0
    passCount1 = 0
    passCount2 = 0

    okPic1 = True
    okPic2 = True
    #tmpPic1 = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
    #tmpPic2 = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
    seperateBLock = np.zeros((webcam_size[1], 60, 3), dtype = "uint8")

    idList = []
    idYN = []
    idScore = []

    '''
    okPic1, pic1 = cam1.takepic(rotate=0, resize=None, savePath=None)
    if(okPic1 is not True):
        print("Taking a picture by cam1 is failed!")
    else:
        tmpPic1 = pic1.copy()

    okPic2, pic2 = cam2.takepic(rotate=0, resize=None, savePath=None)
    if(okPic2 is not True):
        print("Taking a picture by cam2 is failed!")
    else:
        tmpPic2 = pic2.copy()
    '''
    #----------------------
    bbox1 = []
    bbox2 = []

    while (len(bbox1)<1) or (len(bbox2)<1):
        #print(len(bbox1), len(bbox2))
        #bbox1 = []
        #bbox2 = []

        #if(len(bbox1)<1):
        okPic1, pic1 = cam1.takepic(rotate=0, resize=None, savePath=None)
        if(okPic1 is not True):
            print("Taking a picture by cam1 is failed!")
        else:
            tmpPic1 = pic1.copy()
            bbox1 = getFaces_cascade(pic1)
            if(len(bbox1)>0):
                for (x,y,w,h) in bbox1:
                    cv2.rectangle( tmpPic1,(x,y),(x+w,y+h),(0,255,0),2)

        #if(len(bbox2)<1):
        okPic2, pic2 = cam2.takepic(rotate=0, resize=None, savePath=None)
        if(okPic2 is not True):
            print("Taking a picture by cam2 is failed!")
        else:
            tmpPic2 = pic2.copy()
            bbox2 = getFaces_cascade(pic2)
            if(len(bbox2)>0):
                for (x,y,w,h) in bbox2:
                    cv2.rectangle( tmpPic2,(x,y),(x+w,y+h),(0,255,0),2)


        cv2.imshow("SunplusIT", np.hstack((tmpPic1, seperateBLock, tmpPic2)) )
        cv2.waitKey(1)

    #----------------------

    for folderID in os.listdir(validPicPath):
        print("check ",folderID)
        if os.path.exists(validPicPath + folderID + "/cam0/valid.jpg") and \
                os.path.exists(validPicPath + folderID + "/cam1/valid.jpg"):

            valid0 = cv2.imread(validPicPath + folderID + "/cam0/valid.jpg")
            valid1 = cv2.imread(validPicPath + folderID + "/cam1/valid.jpg")

            passYN1, score1 = faceCheck.face_match(face1=pic1, face2=valid0, threshold=0.75)
            passYN2, score2 = faceCheck.face_match(face1=pic2, face2=valid1, threshold=0.75)

            idList.append(folderID)
            idYN.append((passYN1, passYN2))
            idScore.append((score1, score2))

    return idList, idYN, idScore


def matchFace2(employeeID=200334, totalCount=5):
    totalCount1 = 0
    totalCount2 = 0
    passCount1 = 0
    passCount2 = 0

    okPic1 = True
    okPic2 = True
    picSavePath1 = validPicPath + str(employeeID) + "/cam0/"
    picSavePath2 = validPicPath + str(employeeID) + "/cam1/"
    tmpPic1 = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
    tmpPic2 = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
    seperateBLock = np.zeros((webcam_size[1], 60, 3), dtype = "uint8")

    while totalCount1<totalCount or totalCount2<totalCount:

        okPic1, pic1 = cam1.takepic(rotate=0, resize=None, savePath=None)
        if(okPic1 is not True):
            print("Taking a picture by cam1 is failed!")

        okPic2, pic2 = cam2.takepic(rotate=0, resize=None, savePath=None)
        if(okPic2 is not True):
            print("Taking a picture by cam2 is failed!")

        if(okPic1 is True and totalCount1<totalCount):
            bbox = getFaces_cascade(pic1)
            if(len(bbox)>0):
                tmpPic1 = pic1.copy()
                for (x,y,w,h) in bbox:
                    cv2.rectangle( tmpPic1,(x,y),(x+w,y+h),(0,255,0),2)

                valid = cv2.imread(validPicPath + str(employeeID) + "/cam0/valid.jpg")
                passYN1, score1 = faceCheck.face_match(face1=pic1, face2=valid, threshold=0.75)
                totalCount1 += 1
                if(passYN1 is True):
                    passCount1 += 1

        if(okPic2 is True and totalCount2<totalCount):
            bbox = getFaces_cascade(pic2)
            if(len(bbox)>0):
                tmpPic2 = pic2.copy()
                for (x,y,w,h) in bbox:
                    cv2.rectangle( tmpPic2,(x,y),(x+w,y+h),(0,255,0),2)

                valid = cv2.imread(validPicPath + str(employeeID) + "/cam1/valid.jpg")
                passYN2, score2 = faceCheck.face_match(face1=pic2, face2=valid, threshold=0.75)
                totalCount2 += 1
                if(passYN2 is True):
                    passCount2 += 1

        cv2.imshow("SunplusIT", np.hstack((tmpPic1, seperateBLock, tmpPic2)) )
        cv2.waitKey(1)

    return passCount1, passCount2

#------------------------------------------------------------------------
'''
camOK = True
cam1 = webCam(id=0, size=(320,240))
if(cam1.working() is False):
    camOK = False
    print("Web camera #1 is not working!")

cam2 = webCam(id=1, size=(320,240))
if(cam2.working() is False):
    camOK = False
    print("Web camera #2 is not working!")
'''

faceCheck = facenetVerify(graphPath=GRAPH_FILENAME, movidiusID=0)
#------------------------------------------------------------------------

while True:

    idList, idYN, idScore = matchFace()
    if(idList is not None):
        i = 0
        for id in idList:
            print(idList[i], idYN[i], idScore[i])

            i += 1

        time.sleep(6)
