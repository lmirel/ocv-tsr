# import the necessary packages
from threading import Thread
import cv2
from datetime import datetime

class PiFrameSave:
	def __init__(self, **kwargs):
		self.stopped = False
		self.frame_list = []

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		print("#i:start save thread")
		while True:
			if len(self.frame_list) > 0:
				fs = self.frame_list.pop(0)
				if fs is not None:
					# keep looping infinitely until the thread is stopped
					iname = "./raw/thd-image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
					cv2.imwrite (iname, fs)
					print("#i:save frame {}".format(iname))
			# if the thread indicator variable is set, stop the thread
			#if self.stopped:
			#	return
		print("#i:end save thread")

	def stop(self):
		# indicate that the thread should be stopped
		print("#i:stop save thread")
		self.stopped = True

	def count(self):
		# indicate that the thread should be stopped
		return len(self.frame_list)
		
	def save(self, frm):
		# return the frame most recently read
		self.frame_list.append(frm)