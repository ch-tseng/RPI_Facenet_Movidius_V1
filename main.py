#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os, time
import easygui


GRAPH_FILENAME = "facenet_celeb_ncs.graph"
inputType = "webcam"  # webcam, image, video
media = ""
#video_out = "/media/pi/SSD1T/recording/road.avi"
video_out = "record/"

FACE_MATCH_THRESHOLD = 0.75

video_length = 600
framerate = 5.0
webcam_size = (320,240)

validPicPath = "valid/"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_scale = 1.1
cascade_neighbors = 8
minFaceSize = (90,100)

#---------------------------------------------------------
def createEnv():
    if not os.path.exists(validPicPath):
        os.makedirs(validPicPath)
        print("Pics for valid path created:", validPicPath)

    #if not os.path.exists(validPicPath + str(peopleID)):
    #    os.makedirs(validPicPath+str(peopleID))
    #    print("Pics for valid user path created:", validPicPath+str(peopleID))

    #if not os.path.exists(validPicPath + str(peopleID) + "/cam0"):
    #    os.makedirs(validPicPath+str(peopleID) + "/cam0")
    #    print("Pics for valid user path created:", validPicPath+str(peopleID)+"/cam0")

    #if not os.path.exists(validPicPath + str(peopleID) + "/cam1"):
    #    os.makedirs(validPicPath+str(peopleID) + "/cam1")
    #    print("Pics for valid user path created:", validPicPath+str(peopleID)+"/cam1")

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

def run_inference(image_to_classify, facenet_graph):

    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    resized_image = preprocess_image(image_to_classify)

    #cv2.imshow("preprocessed", resized_image)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()

    #print("Total results: " + str(len(output)))
    #print(output)

    return output

def overlay_on_image(display_image, image_info, matching):
    rect_width = 10
    offset = int(rect_width/2)
    if (image_info != None):
        cv2.putText(display_image, image_info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if (matching):
        # match, green rectangle
        cv2.rectangle(display_image, (0+offset, 0+offset),
                      (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
                      (0, 255, 0), 10)
    else:
        # not a match, red rectangle
        cv2.rectangle(display_image, (0+offset, 0+offset),
                      (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
                      (0, 0, 255), 10)

def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image

def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    #print('difference is: ' + str(total_diff))

    if (total_diff < FACE_MATCH_THRESHOLD):
        # the total difference between the two is under the threshold so
        # the faces match.
        print('Pass! difference is: ' + str(total_diff))
        return True, total_diff

    # differences between faces was over the threshold above so
    # they didn't match.
    print('No pass! difference is: ' + str(total_diff))
    return False, total_diff

def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True

def get_cameraimage(id):
    if(id==1):
        hasFrame, infer_image = INPUT1.read()
    else:
        hasFrame, infer_image = INPUT0.read()

    return hasFrame, infer_image

def compare_img(nowImg, orgImg, objImg, camID):
    matching, faceScore = face_match(orgImg, nowImg)

    if(matching==True):
        color = (0,255,0)
        match_text = "#"+str(camID)+" Matched: "+str(round(faceScore,2)) 
    else:
        color = (0,0,255)
        match_text = "#"+str(camID)+" Not matched: "+str(round(faceScore,2))

    print(match_text)
    cv2.putText(objImg, match_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Camera-" + str(camID), objImg)
    #key = cv2.waitKey(1)
    #print("key:", key)

    return matching

def createID(id):
    validated_image_filename0 = validPicPath+str(id)+"/cam0/valid.jpg"
    validated_image_filename1 = validPicPath+str(id)+"/cam1/valid.jpg"
    if not os.path.exists(validPicPath + str(id)):
        os.makedirs(validPicPath+str(id))
        print("Pics for valid user path created:", validPicPath+str(id))

    if not os.path.exists(validPicPath + str(id) + "/cam0"):
        os.makedirs(validPicPath+str(id) + "/cam0")
        print("Pics for valid user path created:", validPicPath+str(id)+"/cam0")

    if not os.path.exists(validPicPath + str(id) + "/cam1"):
        os.makedirs(validPicPath+str(id) + "/cam1")
        print("Pics for valid user path created:", validPicPath+str(peopleID)+"/cam1")

    key = 0
    takingPic = True
    pic_cam0 = False   #has cam0 taken the pic?
    pic_cam1 = False   #has cam1 taken the pic?

    while takingPic:
        for cameraID in [0,1]:
            hasFrame, infer_image = get_cameraimage(cameraID)
            #print("#"+str(cameraID), hasFrame)
            if (not hasFrame):
                easygui.msgbox('相機#'+str(cameraID)+'有問題無法拍照，請洽MIS。')
                takingPic = False
                break

            else:
                displayIMG = infer_image.copy()
                bbox = getFaces_cascade(infer_image)
                print("Faces", len(bbox))
                if(len(bbox)>0):
                    for (x,y,w,h) in bbox:
                        cv2.rectangle( displayIMG,(x,y),(x+w,y+h),(0,255,0),2)

                    if(cameraID==0):
                        valid_pic = validated_image_filename0
                    else:
                        valid_pic = validated_image_filename1

                    if(key==99):
                        print("write:", valid_pic)
                        cv2.imwrite(valid_pic, infer_image) 
                        if(cameraID==0):
                            pic_cam0 = True
                        elif(cameraID==1):
                            pic_cam1 = True

                        if(pic_cam0 is True and pic_cam1 is True):
                            easygui.msgbox('相片拍攝完畢！')
                            takingPic = False
                            break

            cv2.imshow("Camera-"+str(cameraID), displayIMG)

            if(cameraID==1):
                key = cv2.waitKey(1)
                print("key:", key)



start_time = time.time()

INPUT0 = cv2.VideoCapture(0)
INPUT0.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_size[0])
INPUT0.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_size[1])

INPUT1 = cv2.VideoCapture(1)
INPUT1.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_size[0])
INPUT1.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_size[1])

while True:

    ynBox = easygui.ynbox('您想要使用臉孔識別打卡嗎？', '凌陽創新', ('是', '否'))

    if(ynBox):
        peopleID = easygui.integerbox('請輸入您的工號（六碼）：', '工號輸入', lowerbound=200000, upperbound=212000)

        if(peopleID==211111):
            ynBox = easygui.ynbox('您確定要建立臉孔識別打卡的新使用者嗎？', '建立使用者', ('是', '否'))

            if(ynBox):
                peopleID = easygui.integerbox('請輸入新使用者的工號（六碼）：', '工號輸入', lowerbound=200000, upperbound=212000)
                if(chkID(peopleID) is True):
                    easygui.msgbox('已經有此使用者的臉孔資料了！')

                else:
                    ynBox = easygui.ynbox('按下「拍照鍵」五秒後即開始拍照', '凌陽創新', ('拍照', '取消'))
                    if(ynBox):
                        createID(peopleID)

        else:
            if(chkID(peopleID)==True):

                #chkEnv()
                validated_image_filename0 = validPicPath+str(peopleID)+"/cam0/valid.jpg"
                validated_image_filename1 = validPicPath+str(peopleID)+"/cam1/valid.jpg"

                record_time = time.time()

                #------------------------------------------------------
                devices = mvnc.EnumerateDevices()
                if len(devices) == 0:
                    print('No NCS devices found')
                    quit()

                # Pick the first stick to run the network
                device = mvnc.Device(devices[0])

                # Open the NCS
                device.OpenDevice()

                # The graph file that was created with the ncsdk compiler
                graph_file_name = GRAPH_FILENAME

                # read in the graph file to memory buffer
                with open(graph_file_name, mode='rb') as f:
                    graph_in_memory = f.read()

                # create the NCAPI graph instance from the memory buffer containing the graph file.
                graph = device.AllocateGraph(graph_in_memory)

                #--Read the validate image -----------------------------------------------------------

                validated_image0 = cv2.imread(validated_image_filename0)
                validated_image1 = cv2.imread(validated_image_filename1)
                valid_output0 = run_inference(validated_image0, graph)
                valid_output1 = run_inference(validated_image1, graph)
                #----------------------------------------------------------

                key = 0
                while True:
                    if(inputType == "webcam"):

                        for cameraID in [0,1]:
                            hasFrame, infer_image = get_cameraimage(cameraID)

                            #print("#"+str(cameraID), hasFrame)
                            if (not hasFrame):
                                print("#"+str(cameraID)+" Done processing !!!")
                                print("--- %s seconds ---" % (time.time() - start_time))
                                #if(video_out!=""):
                                #    out.release()

                                pass
                            else:
                                test_output = run_inference(infer_image, graph)
                                if(cameraID==0):
                                    valid_pic = valid_output0
                                else:
                                    valid_pic = valid_output1

                                matching = compare_img(test_output, valid_pic, infer_image, cameraID)
                                if(key==99):
                                    picFile = validPicPath+str(peopleID)+"/cam"+str(cameraID)+"/"+str(time.time())+".jpg"
                                    print("write:", picFile)
                                    cv2.imwrite(picFile, infer_image) 

                            if(cameraID==1):
                                key = cv2.waitKey(1)
                                print("key:", key)


