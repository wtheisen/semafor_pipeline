from featureExtraction import featureExtraction
import argparse
import os
import sys
import logging
import traceback
from resources import Resource
import concurrent.futures
from joblib import Parallel, delayed, load, dump
import numpy as np
numJobs = 10
def process_image(input):
    basepath = '/afs/crc.nd.edu/group/cvrl/archive/data/ND/MEDIFOR/'
    checkPath = '/afs/crc.nd.edu/group/cvrl/scratch_4/MFC18_eval/'
    filepath = input.rstrip()
    foldernum = -1
    
    try:
        filename=os.path.basename(filepath);
        #print(filename,' ',filepath)
        worldImageResource= Resource.from_file(filename, filepath)
        relativeLocation = os.path.dirname(filepath)[len(basepath):]
        testOutPath = os.path.join(args.outputdir,relativeLocation,filename)
        if args.checkAgainstDir:
            testCheckPath = os.path.join(args.checkAgainstDir,relativeLocation,filename)
            print('check: ',testCheckPath)
            #print('doing')
        if not os.path.exists(testOutPath) and (args.checkAgainstDir is None or not os.path.exists(testCheckPath)):
            try:
                os.makedirs(os.path.dirname(testOutPath))
            except:
                pass
            print('processing '+filepath);
            print('cores: ', args.TFCores)
            featureDict  = featureExtractor.processImage(worldImageResource,tfcores = args.TFCores,resourcepath=filepath)
            if not foldernum  == -1:
                outputdir = os.path.join(args.outputdir,str(foldernum))
                if not os.path.exists(outputdir):
                    try:
                        os.makedirs(outputdir)
                    except:
                        pass
            else: outputdir = args.outputdir
            outpath = os.path.join(outputdir,os.path.dirname(filepath)[len(basepath):], featureDict['supplemental_information']['value'].key)
            #print outpath
            with open(outpath,'wb') as of:
                of.write(featureDict['supplemental_information']['value']._data)
        else:
            print('skipping ', testOutPath)
    except IOError as e:
        logging.info('skipping '+filepath);
    except Exception as e:
        logging.error(traceback.format_exc())

    return []

parser = argparse.ArgumentParser()
parser.add_argument('--NISTWorldIndex', help='provenance index file')
parser.add_argument('--NISTDataset', help='nist dataset directory')
parser.add_argument('--outputdir', help='output directory')
parser.add_argument('--jobNum', help='job number starting at 0',type=int,default=0)
parser.add_argument('--numJobs', help='total number of jobs',type=int,default=1)
parser.add_argument('--maxFolderSize', help='how many files a folder can hold on the file system',type=int,default=-1)
parser.add_argument('--inputFolder',help='optionally specify a single folder containing images to extract',default=None)
parser.add_argument('--inputFileList',default=None)
parser.add_argument('--checkAgainstDir',default=None)
parser.add_argument('--PE',type=int,default=1)
parser.add_argument('--TFCores',type=int,default=1)
parser.add_argument('--det',help='Keypoint Detection method',default='SURF3')
parser.add_argument('--desc',help='Feature Description method',default='SURF3')
parser.add_argument('--kmax',type=int,default=5000,help='Max keypoints per image')
args = parser.parse_args()
files = []
featureExtractor = featureExtraction(detectiontype=args.det,descriptiontype=args.desc,kmax=args.kmax)
if args.inputFolder is None and args.inputFileList is None:
    with open(args.NISTWorldIndex) as f:
      fileIndex=-1
      lines = f.readlines()
elif args.inputFileList is not None:
    with open(args.inputFileList,'r') as fp:
        lines = fp.readlines()
elif args.inputFolder is not None:
    lines = os.listdir(args.inputFolder)
print('total files: ', len(lines))
partition = len(lines)/args.numJobs
start = int(args.jobNum*partition)
end = int((args.jobNum+1)*partition)

print(start,' ',end)
l1 = lines[0]
if args.inputFolder is None and args.inputFileList is None:
    lines = [l1] + lines[start:end]
else: lines = lines[start:end]
print('partition files: ', len(lines))
if args.maxFolderSize == -1:
    folderNums = np.zeros(len(lines))
folderNums = list(np.arange(start,end)%args.maxFolderSize)

if args.inputFolder is None and args.inputFileList is None:
    for line in lines:
     values = line.split('|')
     if fileIndex==-1:
         fileIndex=values.index('WorldFileName')
     else:
         filepath = os.path.join(args.NISTDataset,values[fileIndex])
         files.append(filepath)
elif args.inputFolder is not None:
     datasetFolder = args.inputFolder
     for line in lines:
         filepath = os.path.join(datasetFolder,line)
         files.append(filepath)
elif args.inputFileList is not None:
    files.extend(lines)
with concurrent.futures.ProcessPoolExecutor(max_workers=args.PE) as executor:
    #for file in files:
    for image_file, output in zip(files, executor.map(process_image, files)):
        print (image_file)
