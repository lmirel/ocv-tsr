#
# https://www.design-reuse.com/articles/41154/traffic-sign-recognition-tsr-system.html
#
# OS needs:
# apt install libleptonica-dev libtesseract-dev tesseract-ocr
# pip3 install tesserocr
#

import numpy as np
import cv2

from datetime import datetime
#from tsrframeocr import TSRFrameOCR
# pip3 install pyserial
import serial

import jetson.utils
import jetson.inference

import argparse
import sys
# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                           formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)
#
try:
    ser = serial.Serial ('/dev/ttyUSB0', 9600, timeout=1)
except:
    print("")
    print ('!serial port NOT accessible')
    sys.exit(0)
# open the serial port
if ser.isOpen ():
    print (ser.name + ' is open...')
else:
    print (ser.name + ' unable to open')
    sys.exit(0)
#
st = 'v'
ser.write (st.encode ())
st = '0000'
ser.write (st.encode ())
st = '    '
ser.write (st.encode ())
#
show_display = True
#
show_fps = True
show_fps = True
#
lFps_sec = 0  #current second
lFps_c = 0    #current fps
lFps_k = 0    #current frames
lFps_M = 0    #max fps
lFps_T = 0    #tot
lFps_rS = 0   #running seconds
cFk = 0       #frame count
#
cs_sec = 0
cs_spd = 0
#
c_r_min = 5 #5 #10
c_r_max = 30 #25 #50

# define range of white color in HSV
sensitivity = 15
lower_white = np.array([0, 0, 255 - sensitivity])
upper_white = np.array([255, sensitivity, 255])
#this is red
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])
#
b_th = 170 #170 #70   #black_threshold 
kFot = 0    #count of saved frames
#
def write_to_7seg (val):
    #never access the serial twice for the same value
    if write_to_7seg._mval == val:
        return
    #
    write_to_7seg._mval = val
    if val == -1:
        st = '    '
    else:
        st = '{:4d}'.format (val)
    #
    ser.write (st.encode ())
#
def check_red_circles (image):
    kTS = "{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    #crop the image area containing the circle
    sub_img = image.copy()
    #1280x720
    c_ry = 180
    c_yy = c_ry
    c_rx = 180
    c_xx = 640
    sub_img = sub_img[c_yy - c_ry:c_yy + c_ry, c_xx - c_rx:c_xx + c_rx]
    # process subimage
    blurred = cv2.blur (sub_img, (5, 5))
    hsv = cv2.cvtColor (blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    # lower mask (0-10)
    #mask0 = cv2.inRange (hsv, lower_white, upper_white)
    mask0 = cv2.inRange (hsv, lower_col1, upper_col1)
    # upper mask (170-180)
    mask1 = cv2.inRange (hsv, lower_col2, upper_col2)
    # join my masks
    cmask = mask0 + mask1
    #cmask = mask1
    #
    result = sub_img
    #cmask = cv2.erode (cmask, None, iterations=2)
    #cmask = cv2.dilate (cmask, None, iterations=2)
    #iname = "./raw/mask-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    #cv2.imwrite (iname, cmask)
    #detect circles
    circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 
                200, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #            60, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #process circles
    c_x = 0
    c_y = 0
    c_r = 0
    if circles is not None:
      kTS = "{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      #iname = "/mnt/raw/img-{}-frame.png".format (kTS)
      #circles = np.uint16 (np.around (circles))
      for i in circles[0,:]:
        c_x = int(i[0])
        c_y = int(i[1])
        c_r = int(i[2]) #autocrop the 'red' circle
        #print("#i:detected circle {}x{}r{}".format(c_x, c_y, c_r))
        if c_r > 6 and c_x > c_r and c_y > c_r:
            tsr_img = sub_img.copy()
            tsr_img = tsr_img[c_y - c_r:c_y + c_r, c_x - c_r:c_x + c_r]
            #
            global kFot
            kFot = kFot + 1
            #cv2.imwrite (iname, final)
            #iname = "./raw/thd-image-{}_{}.png".format (kTS, kFot)
            tsr_imga = cv2.cvtColor (tsr_img, cv2.COLOR_BGR2RGBA)
            cuda_mem = jetson.utils.cudaFromNumpy (tsr_imga)
            #print (cuda_mem)
            class_idx, confidence = net.Classify (cuda_mem, tsr_img.shape[0], tsr_img.shape[1])
            if class_idx >= 0: # or confidence * 100) > 60:
                confi = int (confidence * 1000)
                # find the object description
                class_desc = net.GetClassDesc (class_idx)
                # save images
                iname = "/mnt/raw/img-{}_{}-cuda-{}_{}.jpg".format (kTS, kFot, class_desc, confi)
                jetson.utils.saveImageRGBA (iname, cuda_mem, tsr_img.shape[0], tsr_img.shape[1])
                iname = "/mnt/raw/img-{}_{}-ori.jpg".format (kTS, kFot)
                cv2.imwrite (iname, tsr_img)
                iname = "/mnt/raw/img-{}_{}-frame.jpg".format (kTS, kFot)
                cv2.imwrite (iname, sub_img)
                # overlay the result on the image
                if confi > 990:
                    print ("found sign {} {:s}".format (confi, class_desc))
                    #print ("found sign {} {:s} fps {}".format (confi, class_desc, net.GetNetworkFPS ()))
                    global cs_spd   
                    if class_idx == 0:#kph20    
                        cs_spd = 20 
                    if class_idx == 1:#kph30    
                        cs_spd = 30 
                    if class_idx == 2:#kph50    
                        cs_spd = 50 
                    if class_idx == 3:#kph60    
                        cs_spd = 60 
                    if class_idx == 4:#kph70    
                        cs_spd = 70 
                    if class_idx == 5:#kph80    
                        cs_spd = 80 
                    if class_idx == 6:#kph100   
                        cs_spd = 100    
                    if class_idx == 7:#kph120   
                        cs_spd = 120    
                    global cs_sec   
                    cs_sec = datetime.now().second
                #.if confi > 990:
            #.if class_idx >= 0:
        cv2.circle (result, (c_x, c_y), c_r, (0,0,255), 2)
        #.for
    #.if circles is not None:
    #
    #return tsr_img
    return result
#
ESC=27   
Mm = 0  #max matches

#tsrfocr = TSRFrameOCR ()
#tsrfocr.start ()

# load the recognition network
net = jetson.inference.imageNet (opt.network, sys.argv)

camera = cv2.VideoCapture ('/mnt/cv2video-720p-20191101-140122-097784.avi')
#camera = cv2.VideoCapture ('/mnt/Work/dataset/GOPR1415s.mp4')
#camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GP011416s.mp4')

s_fm = 0 #450  #start frame
write_to_7seg._mval = -2

while True:
    try:
        ret, imgCamColor = camera.read()
        if ret == False:
            print ("#e: read from camera error: {}".format(ret))
            exit(1)
        #
        cFk = cFk + 1
        if imgCamColor is not None:
            result = imgCamColor
            #if cFk % 5 == 0:
            if cFk > s_fm:
              result = check_red_circles (imgCamColor)
            #fps computation
            cFps_sec = datetime.now().second
            lFps_k = lFps_k + 1
            if lFps_sec != cFps_sec:
              lFps_c = lFps_k - 1
              lFps_k = 0
              lFps_rS = lFps_rS + 1 #increment seconds - we assume we get here every second
            if lFps_M < lFps_k:
              lFps_M = lFps_k
            lFps_sec = cFps_sec
            #turn off sign display
            if cs_sec > 0:
                # flash the indicator 
                if cFps_sec % 2 == 0:
                    write_to_7seg (-1)
                else:
                    write_to_7seg (cs_spd)
                # stop flashing and show the last speed
                if cs_sec + 5 < cFps_sec:
                    cs_sec = 0
                    write_to_7seg (cs_spd)
            #
            if show_fps == True:
                #print ("#i:max fps {}".format (lFps_M))
                if lFps_rS > 0:
                  aFps = int (cFk / lFps_rS)
                else:
                  aFps = 1
                cfpst = "FPS M{} / A{} / C{} / T{} p{} s{}".format (lFps_M, aFps, lFps_c, cFk, kFot, lFps_rS)
                #print ("perf {}".format(cfpst))
                #
                if show_display == True:
                    cv2.putText (result, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 0)
                else:
                    if (cFk % 50) == 0:
                        print (cfpst)
            #   
            if show_display == True:
                cv2.imshow ('result', result)
                #
                key = cv2.waitKey (1)
                if key == ESC:
                    break
            #
            #tsrfocr.update()
        else:
            print ("#w:dropping frames {}".format(cFk))
        #
    except KeyboardInterrupt:
        estop = True
        break
#
#tsrfocr.stop()
#print ("#w:dropping {} frames".format (tsrfocr.count()))
#
write_to_7seg (-1)

camera.release()
cv2.destroyAllWindows()
