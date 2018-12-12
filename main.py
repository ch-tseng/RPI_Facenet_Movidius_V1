#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO 
GPIO.setmode(GPIO.BCM)

import logging
from mvnc import mvncapi as mvnc
#import dlib
import numpy as np
import cv2
import imutils
from imutils.face_utils import rect_to_bb
import sys, datetime
import os, time
import easygui
from libFacialDoor import webCam
from libFacialDoor import facenetVerify
import requests
#from libFacialDoor import mqttFACE

#KeyInID = False
runMode = 1  # 0--> enter ID, and add this employee  1--> enter ID and scan all employess to check  2--> enter ID and check only the ID  3--> do not need to enter ID
topDIR ="/media/pi/3A72-2DE1/"
toWebserver = "/var/www/html/door/"
logging.basicConfig(level=logging.INFO, filename=topDIR+'logging.txt')
faceDetect = "cascade"  #dlib / cascade
GRAPH_FILENAME = "facenet_celeb_ncs.graph"
WAV_FOLDER = "/home/pi/works/door_face/wav/"
FACE_MATCH_THRESHOLD_cam0 = 0.40
FACE_MATCH_THRESHOLD_cam1 = 0.40
FACE_MATCH_THRESHOLD_avg = 0.38

#webcam_size = ( 640,360)
webcam_size = ( 352,288)
btnCheckin = 14

offsetFaceBox = (10,10)
captureTime = 60  #how long will camera try to capture the face for verify a face
previewPicPath = topDIR+"preview/"  #all pics face size is not pass the required size
historyPicPath = topDIR+"history/"   #for those face is pass the required size and will be check
validPicPath = topDIR+"valid/"
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
cascade_scale = 1.2
cascade_neighbors = 6
minFaceSize = (140,140)  #for cascade
minFaceSize1 = (180, 180)  #for send to facenet  webcam1
minFaceSize2 = (140, 140)  #for send to facenet webcam2
dlib_detectorRatio = 1

#posturl="http://172.30.8.86/api/DoorFaceDetection"
posturl="http://api.sunplusit.com/api/DoorFaceDetection"
GPIO.setup(btnCheckin, GPIO.IN)
cv2.namedWindow("SunplusIT", cv2.WND_PROP_FULLSCREEN)        # Create a named window
cv2.setWindowProperty("SunplusIT", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

blankScreen = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
seperateBLock = np.zeros((webcam_size[1], 60, 3), dtype = "uint8")
#-----------------------------------------------------------------------

def createEnv():
    if not os.path.exists(validPicPath):
        os.makedirs(validPicPath)
        logging.info("Pics for valid path created:", validPicPath)

    if not os.path.exists(previewPicPath):
        os.makedirs(previewPicPath)
        logging.info("Pics for preview path created:", previewPicPath)
    if not os.path.exists(previewPicPath+"/cam0"):
        os.makedirs(previewPicPath+"/cam0")
        logging.info("Pics for preview path created:", previewPicPath+"/cam0")
    if not os.path.exists(previewPicPath+"/cam1"):
        os.makedirs(previewPicPath+"/cam1")
        logging.info("Pics for preview path created:", previewPicPath+"/cam1")

    if not os.path.exists(historyPicPath):
        os.makedirs(historyPicPath)
        logging.info("Pics for history path created:", historyPicPath)
    if not os.path.exists(historyPicPath+"/cam0"):
        os.makedirs(historyPicPath+"/cam0")
        logging.info("Pics for history path created:", historyPicPath+"/cam0")
    if not os.path.exists(historyPicPath+"/cam1"):
        os.makedirs(historyPicPath+"/cam1")
        logging.info("Pics for history path created:", historyPicPath+"/cam1")

def regID(id, pic1, pic2):
    userPath = validPicPath + id + "/"
    validated_image_filename0 = userPath + "cam0/valid.jpg"
    validated_image_filename1 = userPath + "cam1/valid.jpg"

    if not os.path.exists(userPath):
        os.makedirs(userPath)
    if not os.path.exists(userPath+"cam0"):
        os.makedirs(userPath+"cam0")
    if not os.path.exists(userPath+"cam1"):
        os.makedirs(userPath+"cam1")

    cv2.imwrite(validated_image_filename0, pic1)
    cv2.imwrite(validated_image_filename1, pic2)

    logging.info(id+" path registered:", userPath)

def getFaces_dlib(img):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)
    bboxes = []
    for faceid, rect in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            bboxes.append((x,y,w,h))

    return bboxes

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

def displayScreen(img=None, overlay=None):
    board = cv2.imread("board.png")
    if(img is not None and overlay is not None):
        y_offset = overlay[1]
        x_offset = overlay[0]
        board[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

    return board
    #cv2.imshow("SunplusIT", board )
    #cv2.waitKey(1)

def blackScreen():
    cameraArea = imutils.resize(np.hstack((blankScreen, seperateBLock, blankScreen )), width=800)
    screen = displayScreen(cameraArea, (0,95))
    return screen

#def printText(text=None, color=(0,0,0)):
#    if(text is not None):
#        board = cv2.imread("board.png")

def readNumber(num):
    for i in range(0, len(num)):
        print("play", WAV_FOLDER + "number/" + num[i] + ".wav")
        os.system('/usr/bin/aplay ' + WAV_FOLDER + "number/" + num[i] + ".wav")

def callWebServer(id, pic1, pic2, result):
    filename = str(time.time())

    if(result is True):
        cv2.imwrite(toWebserver + "pass/" + id + "_" + filename + "_cam0.jpg", pic1)
        cv2.imwrite(toWebserver + "pass/" + id + "_" + filename + "_cam1.jpg", pic2)
        print("write to www folder:", toWebserver + "pass/" + id + "_" + filename + "_cam0.jpg")
        url0 = "http://facial-door/door/pass/"+id+"_"+filename+"_cam0.jpg"
        url1 = "http://facial-door/door/pass/"+id+"_"+filename+"_cam1.jpg"
    else:
        cv2.imwrite(toWebserver + "fail/" + id + "_" + filename + "_cam0.jpg", pic1)
        cv2.imwrite(toWebserver + "fail/" + id + "_" + filename + "_cam1.jpg", pic2)
        print("write to www folder:", toWebserver + "fail/" + id + "_" + filename + "_cam0.jpg")
        url0 = "http://facial-door/door/fail/"+id+"_"+filename+"_cam0.jpg"
        url1 = "http://facial-door/door/fail/"+id+"_"+filename+"_cam1.jpg"

    data= {
        'EmpNo': id,
        'FrontFace': url0,
        'SideFace': url1,
        'Detection': result
    }

    logging.info(data)
    logging.info("start posting: {}".format(datetime.datetime.now()))
    r = requests.post(posturl, data=data).text
    logging.info("end posting:{}".format(datetime.datetime.now()))
    logging.info(r)

def matchFace(this_ID=None):
    tmpPic1 = blankScreen.copy()
    tmpPic2 = blankScreen.copy()

    camOK = True
    cam1 = webCam(id=0, size=webcam_size)
    if(cam1.working() is False):
        camOK = False
        logging.error("Web camera #1 is not working!")

    cam2 = webCam(id=1, size=webcam_size)
    if(cam2.working() is False):
        camOK = False
        logging.error("Web camera #2 is not working!")

    if(camOK is not True):
        logging.critical("web camera is not working!")
        return None, None, None

    totalCount1 = 0
    totalCount2 = 0
    passCount1 = 0
    passCount2 = 0

    okPic1 = True
    okPic2 = True
    #tmpPic1 = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
    #tmpPic2 = np.zeros((webcam_size[1], webcam_size[0], 3), dtype = "uint8")
    #seperateBLock = np.zeros((webcam_size[1], 60, 3), dtype = "uint8")

    idList = []
    idYN = []
    idScore = []

    #----------------------
    bbox1 = []
    bbox2 = []
    #Are we get the usable face ?
    faceCam1 = False
    faceCam2 = False

    captureStart = time.time()
    while (faceCam1 is False) or (faceCam2 is False):
        #print(time.time() - captureStart)
        if(time.time() - captureStart > captureTime):
            #print("Time limit")
            screen = displayScreen(None, None)
            
            return (None,None), (None,None), None, None, None, None
            break

        faceCam1 = False
        faceCam2 = False
        okPic1, pic1 = cam1.takepic(rotate=0, vflip=False, hflip=True, resize=None, savePath=None)
        print("pic1:", pic1.shape)
        if(okPic1 is not True):
            logging.error("Taking a picture by cam1 is failed!")
        else:
            tmpPic1 = pic1.copy()
            leftFaceBox = (int(webcam_size[0]/2)-int(minFaceSize[0]/2), int(webcam_size[1]/2)-int(minFaceSize[1]/2))
            rightFaceBox = (int(webcam_size[0]/2)+int(minFaceSize[0]/2), int(webcam_size[1]/2)+int(minFaceSize[1]/2))
            print("leftFaceBox:{}, rightFaceBox:{}".format(leftFaceBox, rightFaceBox))
            cv2.rectangle( tmpPic1,leftFaceBox, rightFaceBox ,(0,0,255),2)
            cv2.putText(tmpPic1, "webcam:0", (int(webcam_size[0]/2)-50, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,0), 1)

            if(faceDetect=='dlib'):
                bbox1 = getFaces_dlib(pic1)
            else:
                bbox1 = getFaces_cascade(pic1)

            centerX = 0
            centerY = 0
            imgCenterX = webcam_size[0] / 2
            imgCenterY = webcam_size[1] / 2
            if(len(bbox1)>0):
                if( bbox1[0][2]>minFaceSize1[0] and bbox1[0][3]>minFaceSize1[1]):
                    centerX = bbox1[0][0] +  bbox1[0][2]/2
                    centerY = bbox1[0][1] +  bbox1[0][3]/2

                    if((centerX<imgCenterX+offsetFaceBox[0] and centerX>imgCenterX-offsetFaceBox[0]) and (centerY<imgCenterY+offsetFaceBox[1] or centerY>imgCenterY-offsetFaceBox[1])):
                        faceCam1 = True
                        faceArea1 = pic1[bbox1[0][1]:bbox1[0][1]+bbox1[0][3],bbox1[0][0]:bbox1[0][0]+bbox1[0][2]]
                        cv2.imwrite(historyPicPath + "cam0/" + str(time.time()) + "-face.jpg", faceArea1)
                        cv2.imwrite(historyPicPath + "cam0/" + str(time.time()) + ".jpg", pic1)
                        logging.debug("write to:", historyPicPath + "cam0/" + str(time.time()) + ".jpg")
                        for (x,y,w,h) in bbox1:
                            cv2.rectangle( tmpPic1,(x,y),(x+w,y+h),(0,255,0),2)

        okPic2, pic2 = cam2.takepic(rotate=180, vflip=False, hflip=True, resize=None, savePath=None)
        print("pic2:", pic2.shape)
        if(okPic2 is not True):
            logging.error("Taking a picture by cam2 is failed!")
        else:
            tmpPic2 = pic2.copy()
            cv2.putText(tmpPic2, "webcam:1", (int(webcam_size[0]/2)-50, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,0), 1)
            #if(faceCam1 == True):
            #    cv2.imwrite(historyPicPath + "cam1/" + str(time.time()) + ".jpg", pic2 )

            faceCam2 = False
            
            if(faceDetect=='dlib'):
                bbox2 = getFaces_dlib(pic2)
            else:
                bbox2 = getFaces_cascade(pic2)

            if(len(bbox2)>0):
                if(bbox2[0][2]>minFaceSize2[0] and bbox2[0][3]>minFaceSize2[1]):
                    faceCam2 = True
                    if(faceCam1 == True):
                        faceArea2 = pic2[bbox2[0][1]:bbox2[0][1]+bbox2[0][3],bbox2[0][0]:bbox2[0][0]+bbox2[0][2]]
                        cv2.imwrite(historyPicPath + "cam1/" + str(time.time()) + "-face.jpg", faceArea2)
                        cv2.imwrite(historyPicPath + "cam1/" + str(time.time()) + ".jpg", pic2 )
                        logging.debug("write to:", historyPicPath + "cam1/" + str(time.time()) + ".jpg")
                    for (x,y,w,h) in bbox2:
                        cv2.rectangle( tmpPic2,(x,y),(x+w,y+h),(0,255,0),2)
            
        #print(tmpPic1.shape, seperateBLock.shape, tmpPic2.shape)
        #tmpPic1 = imutils.resize(tmpPic1, height=280)
        #tmpPic2 = imutils.resize(tmpPic2, height=280)
        #seperateBLock = imutils.resize(seperateBLock, height=280)
        print("tmpPic1:{}, seperateBLock:{}, tmpPic2:{}".format(tmpPic1.shape,seperateBLock.shape,tmpPic2.shape))
        cameraArea = imutils.resize(np.hstack((tmpPic1, seperateBLock, tmpPic2)), width=800)
        screen = displayScreen(cameraArea, (0,95))


        cv2.imshow("SunplusIT", screen)
        cv2.waitKey(1)
    #----------------------
    #Play start to recognize....
    os.system('/usr/bin/aplay ' + WAV_FOLDER + 'recognize.wav')

    if(runMode==1 or runMode==3):
        for folderID in os.listdir(validPicPath):
            if os.path.exists(validPicPath + folderID + "/cam0/valid.jpg") and \
                    os.path.exists(validPicPath + folderID + "/cam1/valid.jpg"):

                valid0 = cv2.imread(validPicPath + folderID + "/cam0/valid.jpg")
                valid1 = cv2.imread(validPicPath + folderID + "/cam1/valid.jpg")

                passYN1, score1 = faceCheck.face_match(face1=faceArea1, face2=valid0, threshold=FACE_MATCH_THRESHOLD_cam0)
                passYN2, score2 = faceCheck.face_match(face1=faceArea2, face2=valid1, threshold=FACE_MATCH_THRESHOLD_cam1)

                idList.append(folderID)
                idYN.append((passYN1, passYN2))
                idScore.append((score1, score2))
                logging.info("ID:{}, PASS1:{}, PASS2:{}, SCORE1:{}, SCORE2:{}".format(folderID, passYN1, passYN2, score1, score2))

    elif(runMode == 2):
        valid0 = cv2.imread(validPicPath + this_ID + "/cam0/valid.jpg")
        valid1 = cv2.imread(validPicPath + this_ID + "/cam1/valid.jpg")
        passYN1, score1 = faceCheck.face_match(face1=faceArea1, face2=valid0, threshold=FACE_MATCH_THRESHOLD_cam0)
        passYN2, score2 = faceCheck.face_match(face1=faceArea2, face2=valid1, threshold=FACE_MATCH_THRESHOLD_cam1)
        idList.append(folderID)
        idYN.append((passYN1, passYN2))
        idScore.append((score1, score2))
        logging.info("ID:{}, PASS1:{}, PASS2:{}, SCORE1:{}, SCORE2:{}".format(folderID, passYN1, passYN2, score1, score2))

    elif(runMode == 0):
        idList = idYN = idScore = None 

    return (pic1,faceArea1), (pic2, faceArea2), idList, idYN, idScore, screen


faceCheck = facenetVerify(graphPath=GRAPH_FILENAME, movidiusID=0)
#------------------------------------------------------------------------

createEnv()

screen = blackScreen()
cv2.imshow("SunplusIT", screen )
cv2.waitKey(1)

while True:
    clickCheckin = GPIO.input(btnCheckin)
    #print(clickCheckin)
    if(clickCheckin == 0):
        logging.info(datetime.datetime.now())
        os.system('/usr/bin/aplay ' + WAV_FOLDER + 'welcome.wav')

        if(runMode == 0 or runMode == 1 or runMode == 2):
            os.system('/usr/bin/aplay ' + WAV_FOLDER + 'inputid.wav')
            peopleID = easygui.integerbox('請輸入您的工號（六碼）：', '工號輸入', lowerbound=200000, upperbound=212000)
            logging.info("User keyin the employee id:", peopleID)

        else:
            peopleID = 0

        (camFace1, faceArea1), (camFace2, faceArea2), idList, idYN, idScore, screen = matchFace(this_ID=peopleID)

        if(idList is not None):
            chkList = []
            i = 0
            for id in idList:
                logging.debug(idList[i], idYN[i], idScore[i])

                if(idYN[i][0] is True and idYN[i][1] is True):
                    chkList.append((idList[i], (idScore[i][0] + idScore[i][1])/2))
                    logging.info("      ---> scores are all pass, added to chkList.")
                i += 1

            logging.info("Final pass list:")
            logging.info(chkList)

            openDoor = False

            if(runMode == 0 or runMode == 1 or runMode == 2):
                #if(len(chkList)>0):
                #os.system('/usr/bin/aplay ' + WAV_FOLDER + 'inputid.wav')
                #peopleID = easygui.integerbox('請輸入您的工號（六碼）：', '工號輸入', lowerbound=200000, upperbound=212000)
                #logging.info("User keyin the employee id:", peopleID)

                if(runMode == 0):  #add the new user
                    filename = str(time.time()) + ".jpg"

                    if(chkID(peopleID) is True):
                        os.rename(validPicPath+str(peopleID)+"/cam0/valid.jpg", validPicPath+str(peopleID)+"/cam0/"+filename)
                        os.rename(validPicPath+str(peopleID)+"/cam1/valid.jpg", validPicPath+str(peopleID)+"/cam1/"+filename)

                    cv2.imwrite(historyPicPath+str(peopleID)+"/cam0/valid.jpg", camFace1)
                    cv2.imwrite(historyPicPath+str(peopleID)+"/cam1/valid.jpg", camFace2)
                    cv2.imwrite(validPicPath+str(peopleID)+"/cam0/valid.jpg", faceArea1)
                    cv2.imwrite(validPicPath+str(peopleID)+"/cam1/valid.jpg"+filename, faceArea2)
                    os.system('/usr/bin/aplay ' + WAV_FOLDER + 'photo_saved.wav')

                elif(runMode == 1 or runMode == 2): 
                    for id, score in chkList:
                        if(int(id) == peopleID and score<FACE_MATCH_THRESHOLD_avg):
                            openDoor = True
                            logging.info("   --->Pass, id is {}, score is {}".format(id, score))


            elif(runMode==3):
                peopleID = 0
                if(len(chkList)>0):
                    smallist = 999
                    for id, (ID, score) in enumerate(chkList):
                        if(score<smallist):
                            smallist = score
                            peopleID = str(ID)

                    if(smallist<FACE_MATCH_THRESHOLD_avg):
                        openDoor = True


            if(openDoor is True):
                logging.info("Send to webserver: peopleID={}, openDoor={}".format(str(peopleID), openDoor))
                callWebServer(str(peopleID), camFace1, camFace2, openDoor)
                cv2.putText(screen, "Your ID is {}, your are verified!".format(peopleID), (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,0,0), 2)
                cv2.imshow("SunplusIT", screen )
                cv2.waitKey(1)
                os.system('/usr/bin/aplay ' + WAV_FOLDER + 'opendoor.wav')
                #readNumber(str(peopleID))
                #os.system('/usr/bin/aplay ' + WAV_FOLDER + 'checkin_opendoor.wav')
            else:
                logging.info("Send to webserver: peopleID={}, openDoor={}".format(str(peopleID), openDoor))
                callWebServer(str(peopleID), camFace1, camFace2, openDoor)
                cv2.putText(screen, "Sorry, you are not verified!", (80, 450), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,255), 2)
                cv2.imshow("SunplusIT", screen )
                cv2.waitKey(1)
                os.system('/usr/bin/aplay ' + WAV_FOLDER + 'sorry_verify_fail.wav')

            screen = blackScreen()
            cv2.imshow("SunplusIT", screen )
            cv2.waitKey(1)

        screen = blackScreen()

        #if(screen is not None):
        cv2.imshow("SunplusIT", screen )
        cv2.waitKey(1)

        #print("Wait 10 seconds")
        #time.sleep(2)
    else:
        cv2.waitKey(1)
