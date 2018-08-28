import numpy as np
import cv2

filename = 'Hackathon2018/QIA-Hackathon\ 2018/Emotion\ Recognition/Dataset/multimodal_video/034-284.mp4'
cap = cv2.VideoCapture(filename)

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('284out.avi', fourcc, 10.0, (640, 480))
face_size = (48,48)
face_out = cv2.VideoWriter('284face.avi', fourcc, 10.0, face_size)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while(cap.isOpened()):
    print("true")
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_color = frame[y:y+h, x:x+w]
            face_color = cv2.resize(roi_color, face_size, interpolation=cv2.INTER_AREA)
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex, ey, ew, eh) in eyes:
            #    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0),2)

        out.write(frame)
        face_out.write(face_color)
       
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
