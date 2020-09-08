import os
import sys
import urllib.request as request
import multiprocessing
import progressbar
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--listFile', help='provenance index file')
parser.add_argument('--outputFolder', help='nist dataset directory')
parser.add_argument('--maxFolderSize', help='directory containing Indexes',type=int,default=-1)
parser.add_argument('--numCores', help='output directory for results',type=int,default=1)
args = parser.parse_args()
outputFolder = args.outputFolder
try:
    os.makedirs(args.outputFolder)
except:
    pass

with open(args.listFile,'r') as fp:
    lines = fp.readlines()

def downloadURL(args):
    url = args[0]
    id = args[1]
    folderNum = args[2]
    if url is not None and id is not None:
        outfolder = os.path.join(outputFolder,folderNum)
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outpath = os.path.join(outfolder, id)
        if not os.path.exists(outpath):
            try:
                request.urlretrieve(url,outpath)
                print(id)
            except:
                pass

inputs = []
bar = progressbar.ProgressBar()
count = 0
maxFolderSize = args.maxFolderSize
for l in bar(lines[1:]):
    parts = l.rstrip().split(',')
    if len(parts) > 1:
        id = parts[0].strip('"')
        url = parts[1][1:-1]
        folderNum = ""
        if maxFolderSize > 0:
            folderNum = str(int(count/maxFolderSize))
        inputs.append((url,id,folderNum))
        count+=1

pool = multiprocessing.Pool(processes=args.numCores)
pool.map(downloadURL,inputs)