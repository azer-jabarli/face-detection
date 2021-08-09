# Face Detection by Azer Jabarli
# 8th August, 2021

import numpy # Necessary for working with .xml files
import cv2 # Necessary for working with cameras, images, videos, etc.

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # Creates face cascade using .xml file

capture = cv2.VideoCapture(0) # Sets the video source to the default camera

while True: # Makes program capture continuously

	ret, frame = capture.read() # Gives the current frame

	frame = cv2.flip(frame, 1) # Mirrors the video 
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts current video frame to grayscale
	faces = face_cascade.detectMultiScale(gray, 1.1, 4) # Gives coordinates of the face


	for (x, y, w, h) in faces: # Draws and moves rectangle around faces
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		grayed = gray[y : y + h, x : x + w]
		colored = frame[y : y + h, x : x + w]

	cv2.imshow("Frame", frame) # Outputs the result in the window

	if cv2.waitKey(1) & 0xFF == ord('q'): # Waits for the key "q" to be pressed
		break

# Releases the capture
capture.release()
cv2.destroyAllWindows()