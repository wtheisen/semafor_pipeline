import numpy as np
import json
import os
import sys
from random import randint
import json
from PIL import Image
import progressbar
trainFilePath = sys.argv[1]
numberOfProbes = int(sys.argv[2])
pathDict = sys.argv[3]
outFileName = sys.argv[4]
with open(pathDict,'r') as fp:
    pdict = json.load(fp)
with open(trainFilePath) as fp:
    lines= fp.readlines()
imageIDs = []
landmarkIDs = []
for l,c in zip(lines[1:],range(len(lines[1:]))):
    l = l.rstrip()
    l = l.replace('"','')
    parts = l.split(',')
    if len(parts) > 2 and not parts[2] == 'None':
        imageIDs.append(parts[0])
        landmarkIDs.append(int(parts[2]))

landmarkIDs = np.array(landmarkIDs)
imageIDs = np.array(imageIDs)
uniqueLandmarkIDs = np.unique(landmarkIDs)
#np.random.shuffle(uniqueLandmarkIDs)
randomProbes = []
outarr  = ['|'.join(['TaskID','ProvenanceProbeFileID','ProvenanceProbeFileName','ProvenanceProbeWidth','ProvenanceProbeHeight'])]
bar = progressbar.ProgressBar()
bad = 0
for uid in bar(uniqueLandmarkIDs[:numberOfProbes]):
    imIDs_unique = imageIDs[landmarkIDs == uid]
    probe = imIDs_unique[randint(0,len(imIDs_unique)-1)]
    if probe in pdict:
        probePath = pdict[probe]
        probePath2 = probePath.split('/')[-2:]
        probePath2 = os.path.join(probePath2[0],probePath2[1])
        randomProbes.append(probe)
        img = Image.open(probePath)
        w = img.size[0]
        h = img.size[1]
        outarr.append('|'.join(['provenance',probe,probePath2,str(w),str(h)]))
    else:
        bad += 1
outstr = '\n'.join(outarr)
print('bad: ',bad)
with open(outFileName,'w') as fp:
    fp.write(outstr)


