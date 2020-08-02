# -*- coding: utf-8 -*-

import os
import lmdb
import cv2
import numpy as np
import glob
import re

def convertFromChars(s):
    s = list(s)

    i = 0
    while True:
        try:
            char = s[i]
            if char.encode("hex") == "c3" or char.encode("hex") == "c5" or char.encode("hex") == "ce" or char.encode("hex") == "c2" or char.encode("hex") == "c6" or char.encode("hex") == "c9":
                s = s[0:i:] + s[i+1: :]
                next_char = s[i]
                if next_char.encode("hex") == "a0" and char.encode("hex") == "c5":
                    s[i] = "cc".decode("hex")
                elif next_char.encode("hex") == "b2" and char.encode("hex") == "c2":
                    s[i] = "cd".decode("hex")
                elif next_char.encode("hex") == "b9" and char.encode("hex") == "c6":
                    s[i] = "ce".decode("hex")
            elif char.encode("hex") == "db" or char.encode("hex") == "d0":
                s = s[0:i+1:] + s[i+2: :]
            elif char.encode("hex") == "e1" or char.encode("hex") == "e2" or char.encode("hex") == "e3":
                s = s[0:i:] + s[i+2: :]
                
                char = s[i]
                if char.encode("hex") == "a0":
                    s[i] = "c1".decode("hex")
                elif char.encode("hex") == "81":
                    s[i] = "c2".decode("hex")
                elif char.encode("hex") == "8f":
                    s[i] = "c3".decode("hex")
                elif char.encode("hex") == "aa":
                    s[i] = "c4".decode("hex")
                elif char.encode("hex") == "a2":
                    s[i] = "c5".decode("hex")
                elif char.encode("hex") == "ac":
                    s[i] = "c6".decode("hex")
                elif char.encode("hex") == "8a":
                    s[i] = "c7".decode("hex")
                elif char.encode("hex") == "8b":
                    s[i] = "c8".decode("hex")
                elif char.encode("hex") == "9c":
                    s[i] = "c9".decode("hex")
                elif char.encode("hex") == "9d":
                    s[i] = "ca".decode("hex")
                elif char.encode("hex") == "98":
                    s[i] = "cb".decode("hex")
        except IndexError:
            break
        i += 1
    s = "".join(s)
    return s

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    outputPath = "../data/Datasets/"
    imagePath = "../data/Latin_images/"

    imagePathList = []
    labelList = []
    
    labelfile = open(imagePath+"smallgt.txt", "r")
    labels = labelfile.read()
    for line in glob.glob(imagePath):
        image = line[:line.find('\n')]
        imagePathList.append(imagePath+image)
        label = labels[labels.find(image[:-3]):]
        label = label[:label.find('\n')]
        label = label[label.replace(',', 'X', 1).find(',')+1:len(label)-1]
        labelList.append(convertFromChars(label))

    createDataset(outputPath, imagePathList, labelList)