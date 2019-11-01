#
import numpy as np
import cv2
from datetime import datetime

lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps

# -*- coding: utf-8 -*-
"""
@author: Javier Perez
@email: javier_e_perez21@hotmail.com

"""


ESC=27   
Mm = 0  #max matches
cFk = 0

#camera = cv2.VideoCapture ('GOPR1415s.MP4')
#camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GP011416s.mp4')
camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GOPR1415s.mp4')

orb = cv2.ORB_create()
bf = cv2.BFMatcher (cv2.NORM_HAMMING, crossCheck=True)

#imgTrainColor = cv2.imread ('train.png')
imgTrainColor = cv2.imread ('speed-30-de.png')
imgTrainGray  = cv2.cvtColor (imgTrainColor, cv2.COLOR_BGR2GRAY)

"""
### create mask
#blur and convert to HSV
blurred = cv2.GaussianBlur (imgTrainColor, (11, 11), 0)
hsv = cv2.cvtColor (blurred, cv2.COLOR_BGR2HSV)
# construct a mask for the color "red", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])
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
#cv2.imwrite ("train-mask.png", cmask)

#cmask = cv2.imread('train-mask.png')
#cmask = cv2.cvtColor (cmask, cv2.COLOR_BGR2GRAY)

#
fw = imgTrainGray.shape[0]
fr = int (fw /2)
# draw mask
mask = np.full (imgTrainGray.shape, 0, dtype=np.uint8)  # mask is only
cv2.circle (mask, (fr, fr), fr, (255, 255, 255), -1)

# get first masked value (foreground)
fg = cv2.bitwise_or (imgTrainGray, imgTrainGray, mask = mask)
# get second masked value (background) mask must be inverted
mask = cv2.bitwise_not (mask)
bg = np.full (imgTrainGray.shape, 255, dtype=np.uint8)
bk = cv2.bitwise_or (bg, bg, mask = mask)
# combine foreground+background
final = cv2.bitwise_or (fg, bk)
"""
kpTrain = orb.detect (imgTrainGray, None)
#kpTrain = orb.detect (imgTrainGray, cmask)
#kpTrain = orb.detect (imgTrainGray, mask)
kpTrain, desTrain = orb.compute (imgTrainGray, kpTrain)

firsttime=True

while True:
  ret, imgCamColor = camera.read()
  cFk = cFk + 1
  if imgCamColor is not None:
    imgCamGray = cv2.cvtColor (imgCamColor, cv2.COLOR_BGR2GRAY)
    kpCam = orb.detect (imgCamGray, None)
    kpCam, desCam = orb.compute (imgCamGray, kpCam)
    matches = bf.match (desCam, desTrain)
    #print ("#ORB matches {}".format(len(matches)))
    dist = [m.distance for m in matches]
    thres_dist = (sum(dist) / len(dist)) * 0.5
    matches = [m for m in matches if m.distance < thres_dist]   
    #print ("#ORB matches {}".format(len(matches)))
    cm = len (matches)
    if Mm < cm:
      Mm = cm

    if firsttime == True:
        h1, w1 = imgCamColor.shape[:2]
        h2, w2 = imgTrainColor.shape[:2]
        nWidth = w1 + w2
        nHeight = max (h1, h2)
        hdif = int((h1 - h2)/2)
        print ("preview size {}x{}".format (nWidth, nHeight))
        firsttime=False
       
    result = np.zeros ((nHeight, nWidth, 3), np.uint8)
    result[hdif:hdif+h2, :w2] = imgTrainColor
    result[:h1, w2:w1+w2] = imgCamColor

    #for i in range(len(matches)):
    #    pt_a=(int(kpTrain[matches[i].trainIdx].pt[0]), int(kpTrain[matches[i].trainIdx].pt[1]+hdif))
    #    pt_b=(int(kpCam[matches[i].queryIdx].pt[0]+w2), int(kpCam[matches[i].queryIdx].pt[1]))
    #    cv2.line(result, pt_a, pt_b, (255, 0, 0))
    #-
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
    cfpst = "FPS {}/{} {}m{} f#{}".format(lFps_M, lFps_c, Mm, cm, cFk)
    cv2.putText (result, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, 0)
        
    cv2.imshow('result', result)
  
    key = cv2.waitKey(1)
    if key == ESC:
        break
camera.release()
cv2.destroyAllWindows()
