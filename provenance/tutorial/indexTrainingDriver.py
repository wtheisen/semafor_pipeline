from indexConstruction import indexConstruction
import argparse
import os
import sys
import logging
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--FeatureFileList', help='provenance index file')
parser.add_argument('--IndexOutputFile', help='output file for the Index Training Parameters')

args = parser.parse_args()

indexConstructor = indexConstruction(cachefolder=os.path.dirname(args.IndexOutputFile),featdims=64)
indexTrainingParameters = indexConstructor.trainIndex(args.FeatureFileList)
outpath = os.path.join(args.IndexOutputFile)
try:
      os.makedirs(os.path.dirname(args.IndexOutputFile))
except:
      pass
with open(outpath,'wb') as of:
      of.write(indexTrainingParameters._data)

