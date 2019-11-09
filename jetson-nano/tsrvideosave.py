#
#

# import the necessary packages

from threading import Thread
import cv2
from datetime import datetime
import time

b_th = 70 #black_threshold 

class TSRvideoSave:
	def __init__(self, **kwargs):
		self.stopped = False
		self.frame_list = []
		self.speed = 0
		self.kFin = 0
		self.kFot = 0

	def start(self, vwriter):
		# start the thread to read frames from the video stream
		t = Thread (target=self.update, args= ())
		t.daemon = True
		t.start ()
		self.vwriter = vwriter
		return self

	def update(self):
		print("#i:start video store thread")
		while self.stopped == False:
			if len (self.frame_list) > 0:
				fs = self.frame_list.pop (0)
				if fs is not None:
					self.kFot = self.kFot + 1
					kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFot)
					#print("#i:VS:process frame {} name {}".format (fs.shape, kTS))
					self.vwriter.write (fs)
			#break
			time.sleep(0.0001)
		print ("#i:end video store thread")

	def stop (self):
		# indicate that the thread should be stopped
		self.stopped = True
		print ("#i:stop video store thread {}>{}/{}".format (self.kFin, self.kFot, len(self.frame_list)))

	def count (self):
		# indicate that the thread should be stopped
		return len(self.frame_list)
		
	def save (self, frm):
		# return the frame most recently read
		self.kFin = self.kFin + 1
		self.frame_list.append (frm)
		kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFin)
		#print ("#i:VS:stored frame {}".format (kTS))  # print ocr text from image
#
