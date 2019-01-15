import cv2

size = (544, 288)

camera0 = cv2.VideoCapture(0)
camera1 = cv2.VideoCapture(1)
camera0.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
camera0.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
camera1.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

def get_pic(id):

    rtnImg = None

    if(id==0):
        if(camera0.isOpened()):
            (grabbed, img) = camera0.read()
            if(grabbed==True):
                rtnImg = img

    elif(id==1):
        if(camera1.isOpened()):
            (grabbed, img) = camera1.read()
            if(grabbed==True):
                rtnImg = img


    return rtnImg

i = 0
while True:

    id = i % 2
    print (id)
    img = get_pic(id)
    if(img is not None):
        print(img.shape)
        cv2.imshow("Frame:{}".format(id), img)
        cv2.waitKey(1)
    i += 1
