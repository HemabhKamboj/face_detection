import numpy as np
import cv2

#Path for face detection xml file
path= "haarcascade_frontalface_default.xml"

#Classifier definition
face_cascade= cv2.CascadeClassifier(path)

#Initialising Webcam video capture
cap = cv2.VideoCapture(0)

while True:
	
	#Reading a single frame from WebCam
	ret,frame= cap.read()
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#Detecting and highlighting faces in the frame
	faces= face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40,40))
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y) ,(x+w,y+h), (255,55,0), 2)

	cv2.imshow("Face detect", frame)
	
	ch= cv2.waitKey(1)
	#Press 'e' to exit
	if ch & 0xFF == ord('e'):
		break

cap.release()
cv2.destroyAllWindows()
