#! /usr/bin/env python3

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os, time

validated_image_filename0 = "validated_images/ch.tseng0.jpg"
validated_image_filename1 = "validated_images/ch.tseng1.jpg"

GRAPH_FILENAME = "facenet_celeb_ncs.graph"
inputType = "webcam"  # webcam, image, video
media = ""
#video_out = "/media/pi/SSD1T/recording/road.avi"
video_out = "record/"

FACE_MATCH_THRESHOLD = 1.2

video_length = 600
framerate = 5.0
webcam_size = (640,480)

#---------------------------------------------------------
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
        return True

    # differences between faces was over the threshold above so
    # they didn't match.
    print('No pass! difference is: ' + str(total_diff))
    return False

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
    matching = False
    if (face_match(orgImg, test_output)):
        matching = True
    else:
        matching = False

    if(matching==True):
        match_text = "#"+str(camID)+" Matched"
    else:
        match_text = "#"+str(camID)+" Not matched"

    cv2.putText(objImg, match_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Camera-" + str(camID), objImg)
    cv2.waitKey(1)

    return matching

start_time = time.time()

if __name__ == "__main__":

    if(inputType == "webcam"):
        INPUT0 = cv2.VideoCapture(0)
        INPUT0.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_size[0])
        INPUT0.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_size[1])

        INPUT1 = cv2.VideoCapture(1)
        INPUT1.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_size[0])
        INPUT1.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_size[1])

        width = webcam_size[0]
        height = webcam_size[1]

    elif(inputType == "image"):
        INPUT = cv2.imread(media)

    elif(inputType == "video"):
        INPUT = cv2.VideoCapture(media)
        width = cv2.CAP_PROP_FRAME_WIDTH

    if(inputType == "image"):
        cv2.imshow("Frame", imutils.resize(INPUT, width=850))

        k = cv2.waitKey(0)
        if k == 0xFF & ord("q"):
            out.release()

    else:
        #if(video_out!=""):
            #width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            #height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

            #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #out = cv2.VideoWriter(video_out + str(time.time()) + ".avi",fourcc, framerate, (int(width),int(height)))

        frameID = 0
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

        while True:
            if(inputType == "webcam"):
                for cameraID in [0,1]:
                    hasFrame, infer_image = get_cameraimage(cameraID)

                    print("#"+str(cameraID), hasFrame)
                    if (not hasFrame):
                        print("#"+str(cameraID)+" Done processing !!!")
                        print("--- %s seconds ---" % (time.time() - start_time))
                        #if(video_out!=""):
                        #    out.release()

                        break

                    test_output = run_inference(infer_image, graph)
                    matching = compare_img(test_output, valid_output0, infer_image, cameraID)
