import numpy as np
import json
import os
import sys
from random import randint
import json
from PIL import Image
import pickle
import progressbar
trainFilePath = sys.argv[1]
outFileName = sys.argv[2]
with open(trainFilePath) as fp:
    lines= fp.readlines()
imageIDs = []
landmarkIDs = []
landmarkForImage = {}
for l,c in zip(lines[1:],range(len(lines[1:]))):
    l = l.rstrip()
    l = l.replace('"','')
    parts = l.split(',')
    if len(parts) > 2 and not parts[2] == 'None':
        imageIDs.append(parts[0])
        landmarkIDs.append(int(parts[2]))
        landmarkForImage[parts[0]]= str(parts[2])

landmarkIDs = np.array(landmarkIDs)
imageIDs = np.array(imageIDs)
uniqueLandmarkIDs = np.unique(landmarkIDs)
#np.random.shuffle(uniqueLandmarkIDs)
bar = progressbar.ProgressBar()

images_for_landmark = {}
for uid in bar(uniqueLandmarkIDs):
    imIDs_unique = imageIDs[landmarkIDs == uid]
    images_for_landmark[str(uid)] = list(imIDs_unique)

outdict = {}
outdict['imageToLandmark'] = landmarkForImage
outdict['landmarkToImages'] = images_for_landmark
with open(outFileName,'w') as fp:
    json.dump(outdict,fp)


