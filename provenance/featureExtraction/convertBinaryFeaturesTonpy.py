import os
import sys
import numpy as np
from resources import Resource
dirIn = sys.argv[1]
dirOut = sys.argv[1]
featureDir = os.path.join(dirOut,'descriptors')
keypointDir = os.path.join(dirOut,'keypoints')
try:
    os.makedirs(featureDir)
except:
    pass
#try:
#    os.makedirs(keypointDir)
#except:
#    pass

def deserializeFeatures(featureResource):
    data = featureResource._data
    data = np.reshape(data[:-4], (int(data[-4]), int(data[-3])))
    return data

for f in os.listdir(dirIn):
    fullpath = os.path.join(dirIn,f)
    if os.path.isfile(fullpath):
        featureResource = Resource(f, np.fromfile(fullpath, 'float32'), 'application/octet-stream')
        features = deserializeFeatures(featureResource)
        print(os.path.join(featureDir,f+'.npy'))
        #meta = features[:, -4:]
        #features = np.ascontiguousarray(features[:, :-4])
        np.save(os.path.join(featureDir,f+'.npy'),features)
        #np.save(os.path.join(keypointDir,f+'.npy'),meta)
