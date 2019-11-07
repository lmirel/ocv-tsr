#
#
"""root@jetson:/home/jetson# jetson_clocks --show
SOC family:tegra210  Machine:NVIDIA Jetson Nano Developer Kit
Online CPUs: 0-1
CPU Cluster Switching: Disabled
cpu0: Online=1 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 IdleStates: WFI=1 c7=1 
cpu1: Online=1 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 IdleStates: WFI=1 c7=1 
cpu2: Online=0 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 IdleStates: WFI=1 c7=1 
cpu3: Online=0 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 IdleStates: WFI=1 c7=1 
GPU MinFreq=76800000 MaxFreq=614400000 CurrentFreq=460800000
EMC MinFreq=204000000 MaxFreq=1600000000 CurrentFreq=1600000000 FreqOverride=0
Fan: speed=0
NV Power Mode: 5W
root@jetson:/home/jetson# jetson_clocks 
root@jetson:/home/jetson# jetson_clocks --show
SOC family:tegra210  Machine:NVIDIA Jetson Nano Developer Kit
Online CPUs: 0-1
CPU Cluster Switching: Disabled
cpu0: Online=1 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 IdleStates: WFI=0 c7=0 
cpu1: Online=1 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 IdleStates: WFI=0 c7=0 
cpu2: Online=0 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 IdleStates: WFI=0 c7=0 
cpu3: Online=0 Governor=schedutil MinFreq=921600 MaxFreq=921600 CurrentFreq=921600 IdleStates: WFI=0 c7=0 
GPU MinFreq=614400000 MaxFreq=614400000 CurrentFreq=614400000
EMC MinFreq=204000000 MaxFreq=1600000000 CurrentFreq=1600000000 FreqOverride=1
Fan: speed=255
NV Power Mode: 5W
--
76800000 vs
614400000
"""
# import the necessary packages

from threading import Thread
import cv2
#from datetime import datetime
import time

b_th = 70 #black_threshold 

class TSRframeSave:
    def __init__(self, **kwargs):
        self.stopped = False
        self.frame_list = []
        self.speed = 0
        self.kFin = 0
        self.kFot = 0
    #
    def start(self):
        # start the thread to read frames from the video stream
        t = Thread (target=self.update, args= ())
        t.daemon = True
        t.start ()
        return self
    #
    def update(self):
        print("#i:start frame save thread")
        while self.stopped == False:
            if len (self.frame_list) > 0:
                fs = self.frame_list.pop (0)
                fn = self.frame_list.pop (0)
                if fs is not None and fn is not None:
                    self.kFot = self.kFot + 1
                    #kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFot)
                    #print("#i:frame save:process frame {} name {}".format (fs.shape, fn))
                    cv2.imwrite (fn, fs)
            #break
            time.sleep(0.0001)
        print ("#i:end save frame thread")
    #
    def stop (self):
        # indicate that the thread should be stopped
        self.stopped = True
        print ("#i:stop save frame thread {}>{}/{}".format (self.kFin, self.kFot, len(self.frame_list)))
    #
    def count (self):
        # indicate that the thread should be stopped
        return len(self.frame_list)
    #
    def save (self, frm, frmfn):
        # return the frame most recently read
        self.frame_list.append (frm)
        self.frame_list.append (frmfn)
        self.kFin = self.kFin + 1
        #kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFin)
        #print ("#i:frame save stored frame {}".format (frmfn))  # print ocr text from image
#