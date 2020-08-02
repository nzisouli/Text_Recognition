# -*- coding: utf-8 -*-
from torchvision import transforms
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import torch
from torch.autograd import Variable
from PIL import Image, ImageFont, ImageDraw
import sys
import glob
from threading import Thread

sys.path.insert(1, 'Text_Recognition/')
import utils
import dataset
import models.crnn as crnn

cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser()
parser.add_argument('--inputPath', required=True, help='path to optinal input video file')
parser.add_argument('--outputPath', required=True, help='path to output frames')
opt = parser.parse_args()

input_path = opt.inputPath
output_path = opt.outputPath

def CRNNProcess(i, j, cropped):
    global east_mean, crnn_mean
    crnn_start = time.time()
    for (image, startX, startY, endY) in cropped:
        # prepare image for text recognition
        image = image.convert('L')
        image = transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        
        image = image.view(1, *image.size())    
        image = Variable(image)
        
        # evaluate image
        model.eval()
        preds = model(image)

        # decode prediction
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        sim_pred = utils.convertToChars(sim_pred)

        # draw text on original image
        img_pil = Image.fromarray(orig[i])
        draw = ImageDraw.Draw(img_pil)
        draw.text((startX, (endY+startY)/2-13), sim_pred, font=font, fill=(255,0,0,255))
        
        orig[i] = np.array(img_pil)

    crnn_perImg[i] += time.time() - crnn_start

    #textBoxes[i] -= 1

def EASTProcess(i, image):
    global east_mean, crnn_mean
    print("processing image " + str(i) + "...")
    east_start = time.time()
    # grab the image dimensions, set the new width and height
    # and then determine the ratio in change 
    #for both the width and height
    (H, W) = image.shape[:2]
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
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
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # get original image for text recognition
    orig_pil = Image.fromarray(orig[i])
    
    # set number of bounding boxes in image, hence number of threads needed
    textBoxes[i] = len(boxes)
    crnn_perImg.append(0)
    cropped = []

    # loop over the bounding boxes
    j = 0
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        cv2.rectangle(orig[i], (startX-5, startY-5), (endX+5, endY+5), (0, 255, 0), 2, shift=0)

        # crop the part of image with text
        cropped.append([orig_pil.crop((startX-20, startY-20, endX+20, endY+20)), startX, startY, endY])

        j += 1

    threads[i] = Thread(target=CRNNProcess, args=(i, j, cropped,))
    threads[i].start()
    east_mean += time.time() - east_start

def saveOutput():
    for i in range(len(orig)):
        while threads[i] == None:
            pass
        while True:
            try:
                threads[i].join()
                break
            except RuntimeError:
                pass

        # save output
        cv2.imwrite(output_path+str(i)+".jpg", orig[i])

# initialize east text detector
east = "Text_Detection/frozen_east_text_detection.pb" #path to input EAST text detector
min_confidence = 0.02 #minimum probability required to inspect a region
width = 704 #resized image width (should be multiple of 32)
height = 704 #resized image height (should be multiple of 32)
(newW, newH) = (width, height)
# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
# load the pre-trained EAST text detector
print("loading EAST text detector...")
net = cv2.dnn.readNet(east)

# initialize CRNN
model_path = 'Text_Recognition/expr/final.pth'
alphabet = "0123456789aáàâäbcdeéèêfghiìîjklmnoóòôöpqrstuúùûüvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?:_@#&+-/%€()'"
alphabet = utils.convertFromChars(alphabet)
# load the pre-trained CRNN model
print("loading CRNN model...")
model = torch.load(model_path)

converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))

font = ImageFont.truetype("Arialbd.ttf", 26)

# load the input images
images = [cv2.imread(file) for file in glob.glob(input_path+"*.jpg")]

# initialize auxialiary tables
orig = images
textBoxes = []
threads = []
for i in range(len(orig)):
    textBoxes.append(-1)
    threads.append(None)

# set time starts
east_mean = 0
crnn_mean = 0
crnn_perImg = []

# begin processing images
start = time.time()

outThread = Thread(target=saveOutput)
outThread.start()

i = 0
for image in images:
    EASTProcess(i, image)
    i += 1

outThread.join()

end = time.time()
totalTime = end - start
fps = totalTime / i
print("Total time: {:.6f} seconds".format(totalTime))
print("Time per image: {:.6f} seconds".format(fps))

for t in crnn_perImg:
    crnn_mean += t

east_mean /= i
crnn_mean /= i

print("EAST Time per image: {:.6f} seconds".format(east_mean))
print("CRNN Time per image: {:.6f} seconds".format(crnn_mean))