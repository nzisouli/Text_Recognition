# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import glob

cv2.ocl.setUseOpenCL(True)

parser = argparse.ArgumentParser()
parser.add_argument('--inputPath', required=True, help='path to optinal input video file')
parser.add_argument('--outputPath', required=True, help='path to output frames')
opt = parser.parse_args()

input_path = opt.inputPath
output_path = opt.outputPath
east = "frozen_east_text_detection.pb" #path to input EAST text detector
min_confidence = 0.02 #minimum probability required to inspect a region
width = 704 #resized image width (should be multiple of 32)
height = 704 #resized image height (should be multiple of 32)

# load the input images
images = [cv2.imread(file) for file in glob.glob(input_path)]
orig = images

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
print()
net = cv2.dnn.readNet(east)

total_start = time.time()
i = -1
for image in images:
    i += 1
    print("[INFO] processing image " + str(i) + "...")
    # grab the image dimensions, set the new width and height
    # and then determine the ratio in change 
    #for both the width and height
    (H, W) = image.shape[:2]
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    
    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))
    print()
    
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]) + 12)
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]) + 10)
            startX = int(endX - w - 15)
            startY = int(endY - h - 12)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    #create vertices for cv2.polyline()
    vertices = []
    for box in boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        point1 = [box[2], box[3]]
        point2 = [point1[0]+10, point1[1]-h]
        point3 = [point1[0]-w, point1[1]-h-40]
        point4 = [point1[0]-w-10, point1[1]+20]

        vertices.append((point1, point2, point3, point4))

    # loop over the bounding vertices
    j = 0
    for vertix in vertices:
        j += 1

        # scale the bounding box coordinates based on the respective
        # ratios
        for point in vertix:
            point[0] = int(point[0] * rW)
            point[1] = int(point[1] * rH)

        if vertix[2][1] < 0:
            vertix[2][1] = 0
        if vertix[3][1] > int(H*newH):
            vertix[3][1] = int(H*newH)
        if vertix[3][0] < 0:
            vertix[3][0] = 0
        if vertix[1][0] > int(W*newW):
            vertix = int(W*newW)

    # loop over the bounding vertices
    for vertix in vertices:
        # draw the bounding box on the image
        cv2.polylines(orig[i], [np.array(vertix)], 1, (0, 255, 0), 2)

    # show the output image
    cv2.imwrite(output_path+str(i)+".jpg", orig[i])
    cv2.waitKey(0)

total_end = time.time()    
# show timing information on text prediction
print("[INFO] total text detection took {:.6f} seconds".format(total_end - total_start))
