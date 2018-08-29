# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
import face_recognition
import time
import subprocess
from os import listdir
from os.path import isfile, join

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
weight = "mmod_human_face_detector.dat"
cnn_detector = dlib.cnn_face_detection_model_v1(weight)
predPath = os.path.join("/mnt/home/qualcomm/2junhan/qk/Preprocessing/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predPath)
fa = FaceAligner(predictor, desiredFaceWidth=256)


ImageDir = "/mnt/home/qualcomm/Hackathon2018/QIA-Hackathon 2018/Emotion Recognition/Dataset/multimodal_video/"
saveDir = "/mnt/home/qualcomm/data/alignedFaceVideo/"
imageFileList = [f for f in listdir(ImageDir) if isfile(join(ImageDir, f))]
imageNum = len(imageFileList)
imageIndex = 0
for imageFile in imageFileList:
    imageIndex += 1
    print("\n\n=========================================================")
    print("image : " + imageFile)
    print("Processing " + str(imageIndex) + " / " + str(imageNum) + "images ... ")
    cap = cv2.VideoCapture(ImageDir + imageFile)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameInterval = frameCount // 10
    print("frame Count is : " + str(frameCount))
    print(frameInterval)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    faceAlignedVid = cv2.VideoWriter(saveDir + "fa_"+imageFile[:-4]+'.avi', fourcc, 10.0, (48, 48))

    frameIndex = 0
    count = 0

    start = time.time()
    skip = False
    while(cap.isOpened()):
        # Capture frame by frame
        ret, frame = cap.read()
        if ret:
            frameIndex += 1
            if frameIndex % frameInterval == 0 or skip:
                if skip:
                    skip = False
                count += 1
                frame = imutils.resize(frame, width=800)
                face_locations = face_recognition.face_locations(frame)
                if len(face_locations) != 0:
                    skip = False
                    print(face_locations)
                    (top, right, bottom, left) = face_locations[0]
                    faceOrig = imutils.resize(frame[top:bottom, left: right], width=200)
                    faceOrig = cv2.resize(faceOrig, (48, 48), interpolation=cv2.INTER_AREA)
                    faceAligned = fa.align(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dlib.rectangle(left=left, top=top, right=right, bottom=bottom))
                    faceAligned = cv2.resize(faceAligned, (48, 48), interpolation=cv2.INTER_AREA)
                    faceAlignedVid.write(faceAligned)
                else:
                    skip = True
        else:
            break
    end = time.time()
    cap.release()
    faceAlignedVid.release()
    print(end - start)
cv2.destroyAllWindows()
print("Processing done!")