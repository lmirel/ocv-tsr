#
import numpy as np
import cv2
from datetime import datetime

lFps_sec = 0
lFps_k = 0
lFps_M = 0

cap = cv2.VideoCapture('GOPR1410.MP4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        #print ("#i:processing frame {} shape {}".format(iname, frame.shape))
	#fps computation
        cFps_sec = datetime.now().second
	lFps_k = lFps_k + 1
        if lFps_sec != cFps_sec:
	    lFps_k = 0
        lFps_sec = cFps_sec
	if lFps_M < lFps_k:
            lFps_M = lFps_k
	    print ("#i:max fps {}".format (lFps_M))

        #cv2.imshow('frame', frame)
	#if cv2.waitKey(1) & 0xFF == ord('q'):
    	#    break

cap.release()
cv2.destroyAllWindows()
