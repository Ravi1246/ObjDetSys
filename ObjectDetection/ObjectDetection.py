import numpy as np
import cv2


# Yolo Loading
netwrk = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
cocoObjs = []    #To store coco objects


with open("coco.names", "r") as f:
    cocoObjs = [line.strip() for line in f.readlines()]  #assigning coco names to cocoObjs

namesOfLayers = netwrk.getLayerNames()   # R G B
output_layers = [
namesOfLayers[i[0] - 1]
for i in netwrk.getUnconnectedOutLayers()
]

# Image Loading
image = cv2.imread("inputimage.jpeg")
height, width, channels = image.shape


# Object Detection
blobFrmImg = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#print(blobFrmImg)
netwrk.setInput(blobFrmImg)
outs = netwrk.forward(output_layers)
#print(outs)


# Screen display information
confidences = []

rectangleBoxes = []

classObjs = []

for output in outs:
    for detection in output:
        getScores = detection[5:]
        objName = np.argmax(getScores)
        confidence = getScores[objName]

        if confidence >= 0.51:
            # When object is detected
            centerX = int(detection[0] * width)
            centerY = int(detection[1] * height)
            wt = int(detection[2] * width)
            ht = int(detection[3] * height)

            # (X, Y)coordinates of rectangle
            coordinateX = int(centerX - wt / 2)
            coordinateY = int(centerY - ht / 2)

            rectangleBoxes.append([coordinateX, coordinateY, wt, ht])   #Append boxes to list

            confidences.append(float(confidence))   #Append confidence values to list

            classObjs.append(objName)   #Append Objects to list

#
indexes = cv2.dnn.NMSBoxes(rectangleBoxes, confidences, 0.5, 0.4)   # NMS Boxes to avoid multiple boxes


for i in range(len(rectangleBoxes)):
    if i in indexes:        #print box for 1 occurences
        coordinateX, coordinateY, wt, ht = rectangleBoxes[i]
        boxLabel = str(cocoObjs[classObjs[i]])

        cv2.rectangle(image, (coordinateX, coordinateY), (coordinateX + wt, coordinateY + ht), (0,0,255), 2) #Rectangle box properties
        cv2.putText(image, boxLabel, (coordinateX, coordinateY + 25), cv2.QT_NEW_BUTTONBAR, 1, (255,0,0), 2)   #Box Label properties

cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()