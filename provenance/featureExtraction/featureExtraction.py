#simple example with color histograms (good for speed, bad for accuracy...)
from rawpy import imread as raw_imread
import featureExtractor
import numpy as np
from scipy import ndimage
from imageio import imread,imsave
#from scipy.misc import imread, imsave
import PIL.Image as Image
import PIL.Image

import cv2
import io

import math
#import pickle

#use the logging class to write logs out
import logging
from resources import Resource
PIL.Image.MAX_IMAGE_PIXELS = None
class featureExtraction:
      #Put variables for your apprach here
      detetype = 'SURF3'
      desctype = 'SURF3'
      kmax = 5000
      #modify these variables
      algorithmName="ND_dsurf_1000_filtering"
      algorithmVersion="1.0"
      featuredims = 32
      #Set any parameteres needed fo the model here
      def __init__(self,detectiontype='SURF3',descriptiontype='SURF3',kmax = 5000):
          self.detetype = detectiontype
          self.desctype = descriptiontype
          self.kmax = kmax
      #extract features from a single image and save to an object
         #input image output object containing
         #worldImage conatains a resource object (includes the filename and data)
      def processImage (self, worldImage,flip=False,downsize=False,tfcores=1,resourcepath=None):

          #extract data
          filename  = worldImage.key
          print('fname: ,',filename)
          imagedata  = worldImage._data
          image = self.deserialize_image(imagedata,resourcepath=resourcepath).astype(np.uint8)
          if downsize or image.shape[0]*image.shape[1] > 16777216*3:
              print('downsizing')
              image = self.downSize(image)
          if image is not None:
              if flip:
                  image = cv2.flip(image,0)
              #extract your features, in this example color histograms
              # print(f'YOOOOO RPATH: {resourcepath}')
              # featuresStruct = featureExtractor.local_feature_detection_and_description(filename, self.detetype, self.desctype, self.kmax, image, mask=None, dense_descriptor=False,
              #                                   default_params=True,tfcores=tfcores)
              featuresStruct = featureExtractor.local_feature_detection_and_description(resourcepath, self.detetype, self.desctype, self.kmax, image, mask=None, dense_descriptor=False,
                                                default_params=True,tfcores=tfcores)
              # print(f'DAT STRUCT: {featuresStruct}')
              features = featuresStruct[1]
              # print(f'FEATS: {features}')
              meta = np.zeros((features.shape[0],4))
              i = 0
              #get location,size,and angle data of all keypoints
              for kp in featuresStruct[0]:
                  d = np.asarray([kp.pt[0],kp.pt[1],kp.size,kp.angle])
                  meta[i] = d.copy()
                  i+=1
              #serialize your features into a Resource object that can be written to a file
              #The resource key should be the input filename
              if not features == [] and features is not None:
                  totalFeatures = features.shape[1]
                  #print(f'META SHAPE: {meta.shape}')
                  features = np.hstack((features,meta))
              else:
                  features = np.zeros((0,0),dtype='float32')
                  totalFeatures = 0
              # print(f'FEATS BEFORE CEREAL: {features}')
              featureResource = Resource(filename, self.serializeFeature(features,w=image.shape[1],h=image.shape[0]), 'application/octet-stream')
              return self.createOutput(featureResource)
          return None

      #creates the API struct
      def createOutput(self,featureResource):
          return {'algorithm': self.createAlgOutput(), 'supplemental_information': self.createFeatureOutput(featureResource)}

      def createAlgOutput(self):
          return  {'name': self.algorithmName.replace(" ", ""), 'version': self.algorithmVersion.replace(" ", "")}

      def createFeatureOutput(self,featureResource):
          return  {'name': 'provenance_features', 'description': 'features extracted for provenance filtering', 'value': featureResource}

      def serializeFeature(self, features,w=-1,h=-1):
          # features = bytes(features[0])
          # dims = np.array([[features]]).shape
          dims = features.shape
          #print(f'DIMS: {dims}')
          # data = np.insert(features,dims[0]*dims[1],np.asarray([dims[0],dims[1],w,h]))
          data = np.insert(features.flatten().astype('float32'),dims[0]*dims[1],np.asarray([dims[0],dims[1],w,h]))
          return data

      def deserialize_image(self,data, flatten=False,resourcepath=None):
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
#      def deserialize_image(self,data, flatten=False):
#          with io.BytesIO(data) as stream:
#             try:
#                img = imread(stream, flatten=flatten)
#                return img
#             except Exception as e:
#                 print('Using rawpy')
#             try:
#                 img = rawpy.imread(stream).postprocess()
#                 return img
#             except Exception as e:
#                 print('error: ',e)
#                 return None


