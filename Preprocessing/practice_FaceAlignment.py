# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
import time
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
weight = "mmod_human_face_detector.dat"
cnn_detector = dlib.cnn_face_detection_model_v1(weight)
predPath = os.path.join("/mnt/home/qualcomm/2junhan/qk/Preprocessing/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predPath)
fa = FaceAligner(predictor, desiredFaceWidth=256)

filename = "/mnt/home/qualcomm/Hackathon2018/QIA-Hackathon 2018/Emotion Recognition/Dataset/multimodal_video/060-226.mp4"
saveDir = "/mnt/home/qualcomm/2junhan/qk/Preprocessing/"

cap = cv2.VideoCapture(filename)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameInterval = frameCount // 10
print("frame Count is : " + str(frameCount))
print(frameInterval)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
faceOrigVid = cv2.VideoWriter(saveDir+'cnn_226face.avi', fourcc, 10.0, (48, 48))
faceAlignedVid = cv2.VideoWriter(saveDir+'cnn_226aligned.avi', fourcc, 10.0, (48, 48))

frameIndex = 0
count = 0

start = time.time()
while(cap.isOpened()):
    print("true")
    # Capture frame by frame
    ret, frame = cap.read()
    if ret:
        frameIndex += 1
        if frameIndex % frameInterval == 0:
            count += 1
            print(count)
            print(frameIndex)
            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = cnn_detector(gray, 2)
            # loop over the face detections
            # for rect in rects:
            for face in rects:
                # extract the ROI of the original face, then align the face
                # using facial landmarks
                #(x, y, w, h) = rect_to_bb(rect)
                x = face.rect.left()
                y = face.rect.top()
                w = face.rect.right() - x
                h = face.rect.bottom() - y
                faceOrig = imutils.resize(frame[y:y + h, x:x + w], width=200)
                faceOrig = cv2.resize(faceOrig, (48, 48), interpolation=cv2.INTER_AREA)
                faceAligned = fa.align(frame, gray, face.rect)
                faceAligned = cv2.resize(faceAligned, (48, 48), interpolation=cv2.INTER_AREA)
            faceOrigVid.write(faceOrig)
            faceAlignedVid.write(faceAligned)
    else:
        break
end = time.time()
cap.release()
faceOrigVid.release()
faceAlignedVid.release()
print(end-start)
cv2.destroyAllWindows()
