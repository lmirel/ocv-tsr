# USAGE
# python picamera_fps_demo.py
# python picamera_fps_demo.py --display 1

# import the necessary packages
from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2
from datetime import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
  help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
  help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

picW = 1280
picH = 720
"""
# initialize the camera and stream
camera = PiCamera()
camera.resolution = (picW, picH)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(picW, picH))
stream = camera.capture_continuous(rawCapture, format="bgr",
  use_video_port=True)

# allow the camera to warmup and start the FPS counter
print("[INFO] sampling frames from `picamera` module...")
time.sleep(2.0)

#fps = FPS().start()
# loop over some frames
for (i, f) in enumerate(stream):
  # grab the frame from the stream and resize it to have a maximum
  # width of 400 pixels
  frame = f.array
  iname = "./raw/pic-image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
  cv2.imwrite (iname, frame)
  print ("#i:saving frame {}".format (iname))
  #print (frame)
  #frame = imutils.resize(frame, width=400)

  # check to see if the frame should be displayed to our screen
  if args["display"] > 0:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

  # clear the stream in preparation for the next frame and update
  # the FPS counter
  rawCapture.truncate(0)
  fps.update()

  # check to see if the desired number of frames have been reached
  if i == 1:
    break
# stop the timer and display FPS information
#fps.stop()
#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
#cv2.destroyAllWindows()
stream.close()
rawCapture.close()
camera.close()
"""
# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from `picamera` module...")
vs = PiVideoStream(resolution=(picW, picH)).start()
time.sleep(2.0)
fps = FPS().start()

frame_list = []
flsize = 0
# loop over some frames...this time using the threaded stream
#while fps._numFrames < args["num_frames"]:
while True:
  # grab the frame from the threaded video stream and resize it
  # to have a maximum width of 400 pixels
  frame = vs.read()
  frame_list.append (frame.copy())
  flsize = flsize + 1
  if flsize == 100:
    #save frames list, each 100 items
    for sf in frame_list:
      iname = "./raw/thd-image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      cv2.imwrite (iname, sf)
      print ("#i:saving frame {}".format (iname))
    frame_list.clear()
    flsize = 0
  #cv2.imwrite (iname, frame)
  #print (frame)
  #frame = imutils.resize(frame, width=400)
  #print ("#i:saving frame")

  # check to see if the frame should be displayed to our screen
  if args["display"] > 0:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

  # update the FPS counter
  fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
