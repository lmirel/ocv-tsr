#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

#
# v4l2-ctl -d /dev/video0 --set-ctrl=bypass_mode=0 --stream-mmap
#

import jetson.inference
import jetson.utils
from jetcam.csi_camera import CSICamera
from jetcam.usb_camera import USBCamera
#
import argparse
import sys
from datetime import datetime
#
import ctypes
import numpy as np
import cv2
#
# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
#
save_video = False
csi_camera = True   # use CSI/USB camera or gstCamera
#
c_r_min = 5 #5 #10
c_r_max = 30 #25 #50

#this is red
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])
#
kFot = 0    #count of saved frames
#
def check_red_circles (image):
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
    result = image
    #
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
        #print ("#i:saving frame {}".format (iname))
        #cv2.imwrite (iname, image)
        if c_r > 6 and c_x > c_r and c_y > c_r:
            #crop the image area containing the circle
            tsr_img = image.copy()
            tsr_img = tsr_img[c_y - c_r:c_y + c_r, c_x - c_r:c_x + c_r]
            #
            global kFot
            kFot = kFot + 1
            # send to OCR engine for interpretation
            tsr_imga = cv2.cvtColor (tsr_img, cv2.COLOR_BGR2RGBA)
            cuda_mem = jetson.utils.cudaFromNumpy (tsr_imga)
            #print (cuda_mem)
            class_idx, confidence = net.Classify (cuda_mem, tsr_img.shape[0], tsr_img.shape[1])
            if class_idx >= 0: # or confidence * 100) > 60:
                confi = int (confidence * 1000)
                if confi > 899:
                    # find the object description
                    class_desc = net.GetClassDesc (class_idx)
                    # save as image
                    iname = "/mnt/raw/img-{}_{}-cuda-{}_{}.jpg".format (kTS, kFot, class_desc, confi)
                    #jetson.utils.saveImageRGBA (iname, cuda_mem, tsr_img.shape[0], tsr_img.shape[1])
                    # overlay the result on the image
                    print ("found sign {} {:s} - net {} fps {}".format (confi, class_desc, net.GetNetworkName (), net.GetNetworkFPS ()))
                    #iname = "/mnt/raw/img-{}_{:.0f}_{}-gray.png".format (kTS, pwp, kFot)
                    #cv2.imwrite (iname, gray)
                    iname = "/mnt/raw/img-{}_{}-ori.jpg".format (kTS, kFot)
                    #cv2.imwrite (iname, tsr_img)
                    iname = "/mnt/raw/img-{}_{}-frame.jpg".format (kTS, kFot)
                    #cv2.imwrite (iname, image)
                    #iname = "./raw/thd-gray-{}_{}.png".format (kTS, kFot)
        # draw the outer circle
        cv2.circle (result, (c_x, c_y), c_r, (0,0,255), 2)
    #return tsr_img
    return result
#
# camera setup
#display = jetson.utils.glDisplay ()
if csi_camera == False:
    camera = jetson.utils.gstCamera (opt.width, opt.height, opt.camera)
    img, width, height = camera.CaptureRGBA (zeroCopy = True)
    jetson.utils.cudaDeviceSynchronize ()
    jetson.utils.saveImageRGBA ("camera.jpg", img, width, height)
    # create a numpy ndarray that references the CUDA memory
    # it won't be copied, but uses the same memory underneath
    aimg = jetson.utils.cudaToNumpy (img, width, height, 4)
    #print (aimg)
    #aimg1 = aimg.astype (numpy.uint8)
    #print ("img shape {}".format (aimg1.shape))
    aimg1 = cv2.cvtColor (aimg, cv2.COLOR_RGBA2BGR)
    #print (aimg1)
    cv2.imwrite ("array.jpg", aimg1)
    # save as image
    #exit()
else:
    #camera = CSICamera (width=opt.width, height=opt.height)
    # or
    camera = USBCamera (width=opt.width, height=opt.height, capture_device=3)
#
# prep video storing
if save_video == True:
    vname = "/mnt/cv2video-{}p-{}.avi".format (opt.height, datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
    #fourcc = cv2.VideoWriter_fourcc(*'X264')  # cv2.VideoWriter_fourcc() does not exist
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # cv2.VideoWriter_fourcc() does not exist
    video_writer = cv2.VideoWriter (vname, fourcc, 30, (opt.width, opt.height))
#
# load the recognition network
net = jetson.inference.imageNet (opt.network, sys.argv)

# create the camera and display
font = jetson.utils.cudaFont ()
# process frames until user exits
#while display.IsOpen():
while True:
    try:
        # capture the image
        if csi_camera == False:
            img, width, height = camera.CaptureRGBA (zeroCopy = True)
            jetson.utils.cudaDeviceSynchronize ()
            # create a numpy ndarray that references the CUDA memory
            # it won't be copied, but uses the same memory underneath
            aimg = jetson.utils.cudaToNumpy (img, width, height, 4)
            #print ("img shape {}".format (aimg1.shape))
            aimg1 = cv2.cvtColor (aimg.astype (np.uint8), cv2.COLOR_RGBA2BGR)
        else:
            aimg1 = camera.read()
        #
        if save_video == True:
            # add frame to video
            video_writer.write (aimg1)
        # do filter and classification
        aimg1 = check_red_circles (aimg1)
        #
        cv2.imshow ("camera", aimg1)
        #net.PrintProfilerTimes ()
        print ("fps {}".format (net.GetNetworkFPS ()))
        # render the image
        #display.RenderOnce (img, width, height)
        # update the title bar
        #display.SetTitle("{:s} | Network {:.0f} FPS".format (net.GetNetworkName (), net.GetNetworkFPS ()))
        #
        cv2.waitKey (1)
    #
    except KeyboardInterrupt:
        break
#
video_writer.release()
cv2.destroyAllWindows()
#
