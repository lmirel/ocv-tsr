import cv2
import numpy as np

from jetcam.csi_camera import CSICamera                                                                                                                                                                                                                                         
from jetcam.usb_camera import USBCamera

#OS side
# sudo mount -t tmpfs -o size=2048M tmpfs /media/ramdisk
#/etc/fstab none /media/ramdisk tmpfs nodev,nosuid,noexec,nodiratime,size=2048M 0 0
# https://www.techrepublic.com/article/how-to-use-a-ramdisk-on-linux/

import jetson.utils
import argparse


# parse the command line
parser = argparse.ArgumentParser()

parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0), or for VL42 cameras the /dev/video node to use (e.g. /dev/video0).  By default, MIPI CSI camera 0 will be used.")

opt = parser.parse_args()
print(opt)

# create display window
#display = jetson.utils.glDisplay()

# create camera device
#camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
#camera = jetson.utils.gstCamera(1280, 720, "/dev/video3")
# open the camera for streaming
#camera.Open()

# capture frames until user exits
#while display.IsOpen():
#    image, width, height = camera.CaptureRGBA()
#    display.RenderOnce(image, width, height)
#    display.SetTitle("{:s} | {:d}x{:d} | {:.0f} FPS".format("Camera Viewer", width, height, display.GetFPS()))
    
#FPS computation
from datetime import datetime
lFps_sec = 0 #current second
lFPSbeg = 0  #the seccond we started to run
lFPSrun = 0  #running for X secconds
lFPSfnm = 0  #number of frames
#
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps
use_display = True
# Create a VideoCapture object
#capture = cv2.VideoCapture(0)
#capture = CSICamera (width=1280, height=720)                                                                                                                                                                                                                                     
camera = USBCamera (width=1280, height=720, capture_width=1280, capture_height=720, capture_device=3)

# Check if camera opened successfully
#if (capture.isOpened() == False): 
#  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
w   = 1280 #int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
h   = 720  #int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))
fps = 60
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 9FPS
#fourcc = cv2.VideoWriter_fourcc(*'X264')  # 6FPS
fourcc = cv2.VideoWriter_fourcc(*'MJPG')   # 20FPS
video_writer = cv2.VideoWriter("/mnt/Temp/output.avi", fourcc, 30, (w, h))
#video_writer = cv2.VideoWriter("/media/ramdisk/output.avi", fourcc, 30, (w, h))
# record video
lFPSbeg = datetime.now()
while True:
  try:
    frame = camera.read()
    #frame, width, height = camera.CaptureRGBA (zeroCopy=1)
    #frame = jetson.utils.cudaToNumpy (frame, width, height, 4)  # type: np.ndarray
    video_writer.write (frame)
    #
    #fps computation
    cts = datetime.now()
    lFPSrun = (cts - lFPSbeg).seconds * 1000 + (cts - lFPSbeg).microseconds / 1000
    lFPSfnm = lFPSfnm + 1
    if lFPSrun > 0:
      FPSavg = lFPSfnm * 1000 / lFPSrun
    else:
      FPSavg = 0
    cfpst = "FPS: %d t: %dsec %.2ffps"%(lFPSfnm, int(lFPSrun/1000), FPSavg)
    if use_display:
      if True or (lFPSfnm % 10) == 0:
        cv2.putText (frame, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video Stream', frame)
    else:
      if (lFPSfnm % 50) == 0:
        print (cfpst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  except KeyboardInterrupt:
    break
#
# close the camera
camera.Close()

#capture.release()
video_writer.release()
cv2.destroyAllWindows()
