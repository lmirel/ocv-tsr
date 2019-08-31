#
import numpy as np
import cv2
from datetime import datetime

lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps


camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GOPR1415s.mp4')
cFk = 0
ESC = 27
while(camera.isOpened()):
    ret, frame = camera.read()
    if frame is not None:
        cFk = cFk + 1
        result = frame
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        #print ("#i:processing frame {} shape {}".format(iname, frame.shape))
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
        cfpst = "FPS {}/{} f#{}".format(lFps_M, lFps_c, cFk)
        cv2.putText (result, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, 0)

        cv2.imshow('result', result)
  
        key = cv2.waitKey(1)
        if key == ESC:
            break
    else:
      print ("#w:dropping frames {}".format(cFk))

camera.release()
cv2.destroyAllWindows()
