# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils import face_utils
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
import argparse

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
weight = "mmod_human_face_detector.dat"
cnn_detector = dlib.cnn_face_detection_model_v1(weight)
parser = argparse.ArgumentParser(description='PyTorch FinalTest CNN Test')

parser.add_argument('--input_path', type=str, default="/mnt/home/qualcomm/junsu/qk/FinalTestData/", help='input data path')
parser.add_argument('--output_path', type=str, default="/mnt/home/qualcomm/junsu/qk/FinalTestData/FaceAlignedVideo/", help='output data path')

opt = parser.parse_args()


predPath = os.path.join("/mnt/home/qualcomm/junsu/qk/Preprocessing/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predPath)
fa = FaceAligner(predictor, desiredFaceWidth=256)


ImageDir_FinalTest = opt.input_path
saveDir = opt.output_path

print("\n\n===========process for FinalTest==================")
imageFileList = [f for f in listdir(ImageDir_FinalTest) if isfile(join(ImageDir_FinalTest, f))]
imageNum = len(imageFileList)
imageIndex = 0
for imageFile in imageFileList:
    imageIndex += 1
    print("\n\n=========================================================")
    print("image : " + imageFile)
    print("Processing " + str(imageIndex) + " / " + str(imageNum) + "images ... ")
    cap = cv2.VideoCapture(ImageDir_FinalTest + imageFile)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameInterval = frameCount // 3
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
                    print("use cnn detector")
                    face_size = (48, 48)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rects = cnn_detector(gray, 2)

                    # rects = detector(gray, 2)
                    for face in rects:
                        # extract the ROI of the original face, then align the face
                        # using facial landmarks
                        # (x, y, w, h) = rect_to_bb(rect)
                        x = face.rect.left()
                        y = face.rect.top()
                        w = face.rect.right() - x
                        h = face.rect.bottom() - y
                        print(x)
                        print(y)
                        print(w)
                        print(h)

                        faceAligned = fa.align(frame, gray, face.rect)
                        faceAligned = cv2.resize(faceAligned, (48, 48), interpolation=cv2.INTER_AREA)
                    faceAlignedVid.write(faceAligned)
                    skip = False
                else:
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
                        print("cannot use face_recognition")
                        skip = True
            else:
                pass
        else:
            break
    end = time.time()
    cap.release()
    faceAlignedVid.release()
    print(end - start)

cv2.destroyAllWindows()
print("Processing done!")
