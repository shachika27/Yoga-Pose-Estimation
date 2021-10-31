import cv2
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

protoFile = "pose_deploy_linevec_faster_4_stages.prototxt.txt"
weightsFile = "pose_iter_160000.caffemodel"

nPoints = 15

POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

inWidth = 368
inHeight = 368

threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

model = load_model('model.h5')
pic='archive/DATASET/test3.jpg'
# cv2.imshow('Hello',pic)
# cv2.waitKey(0)
frame = cv2.imread(pic)
frameWidth, frameHeight = frame.shape[1], frame.shape[0]
img = np.zeros([frameHeight, frameWidth, 3], dtype=np.uint8)

img.fill(255) #filling with white color
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob) #giving the image as an input to the network
output = net.forward() #receiving output
H, W = output.shape[2], output.shape[3]
points = []
for i in range(nPoints):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H
    if prob > threshold:
        points.append((int(x), int(y)))
    else:
        points.append(None)
if points[14]:
    center = (frame.shape[1]//2, frame.shape[0]//2)
    shift = np.subtract(points[14], center)
    newPoints = [(0,0) for _ in range(len(points))]
    for i in range(len(points)):
        if points[i]:
            newPoints[i] = tuple(np.subtract(points[i], shift))
        else:
            newPoints[i] = None
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if newPoints[partA] and newPoints[partB]:
            cv2.line(img, newPoints[partA], newPoints[partB], (0, 0, 0), 2)
            

dispimg=img
img=cv2.resize(img,(200,200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
classes=['DownDog','Goddess','Plank','Tree','Warrior2']
arr=model.predict(x)
res = np.argmax(arr)
font = cv2.FONT_HERSHEY_SIMPLEX
dispimg=cv2.resize(dispimg,(500,500))
cv2.putText(dispimg,'Pose detected : '+classes[res],(0,40), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
frame=cv2.resize(frame,(500,500))
cv2.imshow('Yoga pose Original',frame)
cv2.imshow('Yoga pose Skeleton',dispimg)
cv2.waitKey(0)
