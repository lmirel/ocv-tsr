#
import numpy as np
import cv2
from datetime import datetime

lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps

def check_red_circles(image):
    blurred = cv2.GaussianBlur (image, (11, 11), 0)
    hsv = cv2.cvtColor (blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    # lower mask (0-10)
    mask0 = cv2.inRange (hsv, lower_col1, upper_col1)
    # upper mask (170-180)
    mask1 = cv2.inRange (hsv, lower_col2, upper_col2)
    # join my masks
    cmask = mask0 + mask1
    #
    cmask = cv2.erode (cmask, None, iterations=2)
    cmask = cv2.dilate (cmask, None, iterations=2)
    #iname = "./raw/mask-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    #cv2.imwrite (iname, cmask)
    #detect circles
    circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 60,
              param1=100, param2=20, minRadius=30, maxRadius=200)
    #process circles
    c_x = 0
    c_y = 0
    c_r = 0
    if circles is not None:
      #iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      #cv2.imwrite (iname, image)
      #print ("#i:saving frame {}".format (iname))
      #circles = np.uint16 (np.around (circles))
      for i in circles[0,:]:
        c_x = int(i[0])
        c_y = int(i[1])
        c_r = int(i[2])
        #print("#i:detected circle {}x{}r{}".format(c_x, c_y, c_r))
        # draw the outer circle
        cv2.circle (image, (c_x, c_y), c_r, (0,255,0), 2)
            
orb = cv2.ORB()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

imgTrainColor=cv2.imread('train.png')
imgTrainGray = cv2.cvtColor(imgTrainColor, cv2.COLOR_BGR2GRAY)

kpTrain = orb.detect(imgTrainGray,None)
kpTrain, desTrain = orb.compute(imgTrainGray, kpTrain)

firsttime=True

while True:
   
    ret, imgCamColor = camera.read()
    imgCamGray = cv2.cvtColor(imgCamColor, cv2.COLOR_BGR2GRAY)
    kpCam = orb.detect(imgCamGray,None)
    kpCam, desCam = orb.compute(imgCamGray, kpCam)
    matches = bf.match(desCam,desTrain)
    dist = [m.distance for m in matches]
    thres_dist = (sum(dist) / len(dist)) * 0.5
    matches = [m for m in matches if m.distance < thres_dist]   

    if firsttime==True:
        h1, w1 = imgCamColor.shape[:2]
        h2, w2 = imgTrainColor.shape[:2]
        nWidth = w1+w2
        nHeight = max(h1, h2)
        hdif = (h1-h2)/2
        firsttime=False
       
    result = np.zeros((nHeight, nWidth, 3), np.uint8)
    result[hdif:hdif+h2, :w2] = imgTrainColor
    result[:h1, w2:w1+w2] = imgCamColor

    for i in range(len(matches)):
        pt_a=(int(kpTrain[matches[i].trainIdx].pt[0]), int(kpTrain[matches[i].trainIdx].pt[1]+hdif))
pt_b=(int(kpCam[matches[i].queryIdx].pt[0]+w2), int(kpCam[matches[i].queryIdx].pt[1]))
        cv2.line(result, pt_a, pt_b, (255, 0, 0))

    cv2.imshow('Camara', result)

cap = cv2.VideoCapture('/home/jetson/Work/dataset/GOPR1412.MP4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        #print ("#i:processing frame {} shape {}".format(iname, frame.shape))
        check_red_circles(frame)
        #fps computation
        cFps_sec = datetime.now().second
        lFps_k = lFps_k + 1
        if lFps_sec != cFps_sec:
          lFps_c = lFps_k - 1
          lFps_k = 0
        if lFps_M < lFps_k:
          lFps_M = lFps_k
        lFps_sec = cFps_sec
          #print ("#i:max fps {}".format (lFps_M))
        cfpst = "FPS {}/{}".format(lFps_M, lFps_c)
        cv2.putText (frame, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        #
        cv2.imshow ('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
