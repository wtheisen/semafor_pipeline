from rawpy import imread as raw_imread
import featureExtractor
import numpy as np
from scipy import ndimage
from imageio import imread,imsave
import PIL.Image as Image
import PIL.Image

import cv2
import io

import math

from resources import Resource
PIL.Image.MAX_IMAGE_PIXELS = None

class PreFeatureExtraction:
    alg_name = "mews_pipeline"
    alg_num = "0.1"
    detetype = 'SURF3'
    desctype = 'SURF3'
    kmax = 5000
    featuredims = 32

    def __init__(self, det='SURF3', desc='SURF3', kmax = 5000):
        self.det_type = det
        self.desc_type = desc
        self.kmax = kmax

    def process_image(self, flip=False, downsize=False, tfcores=1, resource_path=None):
        fname  = worldImage.key
        image_data  = worldImage._data
        image = self.deserialize_image(image_data, resource_path = resource_path).astype(np.uint8)

        if downsize and image is not None:
            print('[LOG]: Downsizing image...')
            image = self.down_size(image)
        if flip and image is not None:
            print('[LOG]: Flipping image...')
            image = cv2.flip(image, 0)

        try:
            features_struct = featureExtractor.feature_detection_and_description(resource_path, self.det_type,
                    self.desc_type, self.kmax, image, tfcores=tfcores)

            features = features_struct[1]
            meta = np.zeros((features.shape[0],4))

            i = 0
            #get location,size,and angle data of all keypoints
            for kp in features_struct[0]:
                d = np.asarray([kp.pt[0], kp.pt[1], kp.size, kp.angle])
                meta[i] = d.copy()
                i += 1

            #serialize your features into a Resource object that can be written to a file
            #The resource key should be the input filename
            if not features == [] and features is not None:
                total_features = features.shape[1]
                features = np.hstack((features, meta))
            else:
                features = np.zeros((0, 0), dtype='float32')
                total_features = 0

            feature_resource = Resource(fname, self.serialize_features(features, w=image.shape[1], h=image.shape[0]), 'application/octet-stream')

            return self.create_output(feature_resource)

        except:
            print("[WARNING]: Feature extraction failure for {fname}")

        return None

    #creates the API struct
    def create_output(self, feature_resource):
        return {'algorithm': self.create_alg_output(), 'supplemental_information': self.create_feature_output(feature_resource)}

    def createAlgOutput(self):
        return  {'name': self.alg_name.replace(" ", ""), 'version': self.alg_num.replace(" ", "")}

    def createFeatureOutput(self, feature_resource):
        return  {'name': 'provenance_features', 'description': 'features extracted for provenance filtering', 'value': feature_resource}

    def serialize_features(self, features, w=-1, h=-1):
        dims = features.shape
        data = np.insert(features.flatten().astype('float32'), dims[0] * dims[1], np.asarray([dims[0], dims[1], w, h]))
        return data

    def deserialize_image(self, data, flatten=False,resourcepath=None):
        fail = False
        try:
            imageStream = io.BytesIO()
            imageStream.write(data)
            imageStream.seek(0)
            imageBytes = np.asarray(bytearray(imageStream.read()), dtype=np.uint8)
            img = cv2.imdecode(imageBytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(e)
            fail = True

        if not fail:
            return img

        with io.BytesIO(data) as stream:
            fail = False
            try:
                try:
                    img = imread(stream, flatten=False)
                except:
                    img = Image.load(stream)
                try:
                    img.mode
                    print('image mode: ',img.mode)
                    img = np.array(img)
                except:
                    print('no image mode')
                    try:
                        print('image object: ',img)
                        print('image shape:')
                        print(img.shape)
                        img.shape[0]
                    except:
                        print('no image shape')
                        img = np.array(imread(stream).reshape(1)[0])
            except Exception as e:
                fail = True
                print('failed scipy')
                print(e)
            if not fail:
                print('Read image with scipy.')
                return img

            fail = False
            try:
                print('read image with rawpy')
                if resourcepath is not None:
                    img = raw_imread(resourcepath).postprocess()
                else:
                    img = raw_imread(stream).postprocess()
            except Exception as e:
                fail = True
                print(e)
            if not fail:
                print('Read image with rawpy.')
                return img

        print('Could not read image')
        return None

    def downSize(self,img):
        maxPixelCount = 2073600  # HDV

        newHeight = 0
        newWidth = 0

        if img.shape[0] * img.shape[1] > maxPixelCount:
            aspectRatio = img.shape[0] * pow(img.shape[1], -1)

            newWidth = int(round(math.sqrt(maxPixelCount / aspectRatio)))
            newHeight = int(round(aspectRatio * newWidth))

            return cv2.resize(img, (newWidth, newHeight))
        else:
            return img
