# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predPath = os.path.join("/mnt/home/qualcomm/2junhan/qk/Preprocessing/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predPath)
fa = FaceAligner(predictor, desiredFaceWidth=256)

filename = "/mnt/home/qualcomm/Hackathon2018/QIA-Hackathon 2018/Emotion Recognition/Dataset/multimodal_video/060-226.mp4"
saveDir = "/mnt/home/qualcomm/2junhan/qk/Preprocessing/"

cap = cv2.VideoCapture(filename)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
faceOrigVid = cv2.VideoWriter(saveDir+'n_226face.avi', fourcc, 10.0, (640, 480))
faceAlignedVid = cv2.VideoWriter(saveDir+'n_226aligned.avi', fourcc, 10.0, (640, 480))

while(cap.isOpened()):
    print("true")
    #Capture frame by frame
    ret, frame = cap.read()
    if ret:
        print("1")
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        # loop over the face detections
        for rect in rects:
            # extract the ROI of the original face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(frame[y:y+h, x:x+w], width=200)
            faceOrig = cv2.resize(faceOrig, (640, 480), interpolation=cv2.INTER_CUBIC)
            faceAligned = fa.align(frame, gray, rect)
        faceOrigVid.write(faceOrig)
        faceAlignedVid.write(faceAligned)
    else:
        break
cap.release()
faceOrigVid.release()
faceAlignedVid.release()
cv2.destroyAllWindows()
