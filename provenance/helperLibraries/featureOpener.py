from resources import Resource
import os
import numpy as np

def openFeaturefile(fileName,returnMetadata = False):
    featuresResource = Resource(os.path.basename(fileName), np.fromfile(fileName, 'float32'), 'application/octet-stream')
    features, otherProps = deserializeFeatures(featuresResource)
    meta = features[:,40:]
    features = np.ascontiguousarray(features[:,:40])
    if returnMetadata:
        return features,meta
    return features
def deserializeFeatures(featureResource):
    data = featureResource._data
    image_width = data[-2]
    image_height = data[-1]
    #print('feature size: ',int(data[-4]),' ',int(data[-3]))
    data = np.reshape(data[:-4],(int(data[-4]),int(data[-3])))
    return data,(image_width,image_height)