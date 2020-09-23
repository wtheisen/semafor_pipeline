print("starting everything")

from indexConstruction import indexConstruction
import argparse
import os
import sys
import logging
import traceback

print("starting things")

parser = argparse.ArgumentParser()
parser.add_argument('--FeatureFileList', help='provenance index file')
parser.add_argument('--IndexOutputFile', help='output file for the Index Training Parameters')

args = parser.parse_args()

indexConstructor = indexConstruction(cachefolder=os.path.dirname(args.IndexOutputFile),featdims=20)
indexTrainingParameters = indexConstructor.trainIndex(args.FeatureFileList)
outpath = os.path.join(args.IndexOutputFile)
try:
      os.makedirs(os.path.dirname(args.IndexOutputFile))
except:
      pass
print(outpath)
with open(outpath,'wb') as of:
      of.write(indexTrainingParameters._data)

