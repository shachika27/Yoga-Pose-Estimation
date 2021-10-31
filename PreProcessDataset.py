from glob import glob
import cv2
import time

downdog = glob("archive/DATASET/TEST/downdog/*")
goddess = glob("archive/DATASET/TEST/goddess/*")
plank = glob("archive/DATASET/TEST/plank/*")
tree = glob("archive/DATASET/TEST/tree/*")
warrior2 = glob("archive/DATASET/TEST/warrior2/*")
classes = [downdog, goddess, plank, tree, warrior2]

protoFile = "pose_deploy_linevec_faster_4_stages.prototxt.txt"
weightsFile = "pose_iter_160000.caffemodel"

nPoints = 15

POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

inWidth = 368
inHeight = 368

from datetime import datetime
import numpy as np
import cv2

startTime = datetime.now()
n = 0
c=0
arr=['downdog','goddess','plank','tree','warrior2']
for pose_class in classes:
    for pic in pose_class:
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
            n+=1
            print(n)
            
            print('archive/DATASET/test_processed/'+arr[c]+'_processed/'+str(n)+'.jpg')
            cv2.imwrite('archive/DATASET/test_processed/'+arr[c]+'_processed/'+str(n)+'.jpg', img)
           
            if(n==1):
                exit
    c=c+1
      