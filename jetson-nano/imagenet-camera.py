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

import argparse
import sys

import ctypes
import numpy
import cv2

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

camera = jetson.utils.gstCamera (opt.width, opt.height, opt.camera)
#display = jetson.utils.glDisplay ()

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
print (aimg1)
cv2.imwrite ("array.jpg", aimg1)
# save as image

#exit()
#
# video
from datetime import datetime
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
        img, width, height = camera.CaptureRGBA (zeroCopy = True)
        jetson.utils.cudaDeviceSynchronize ()
        # create a numpy ndarray that references the CUDA memory
        # it won't be copied, but uses the same memory underneath
        aimg = jetson.utils.cudaToNumpy (img, width, height, 4)
        #print ("img shape {}".format (aimg1.shape))
        aimg1 = cv2.cvtColor (aimg.astype (numpy.uint8), cv2.COLOR_RGBA2BGR)
        # add frame to video
        video_writer.write (aimg1)
        #
        cv2.imshow ("camera", aimg1)
        # classify the image
        class_idx, confidence = net.Classify (img, width, height)
        if (confidence * 100) > 60:
            # find the object description
            class_desc = net.GetClassDesc (class_idx)
            # overlay the result on the image
            font.OverlayText(img, width, height, "{:05.2f}% {:s}".format (confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
            print ("found {:05.2f}% {:s} - net {} fps {}".format (confidence * 100, class_desc, net.GetNetworkName (), net.GetNetworkFPS ()))
            # print out performance info
            net.PrintProfilerTimes ()
        print ("- net {} fps {}".format (net.GetNetworkName (), net.GetNetworkFPS ()))
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