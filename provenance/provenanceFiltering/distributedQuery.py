#this class will be modified by the MediFor Compute team (It should not be touched by TA1 provenance performers)
#It provides client stubs that can be called for provenance filtering that are guarenteed to
#run against all indices

#eventually the hope is to load the entire index into RAM to minimize disk read / write by wrapping these using MESOS / Docker
from featureExtraction import featureExtraction
import numpy as np
import pickle
import traceback
import sys
from resources import Resource
from queryIndex import  queryIndex,filteringResults
from fileutil import make_sure_path_exists
import multiTierFeatureBuilder
import os
from joblib import Parallel,delayed
import progressbar
import scoreMerge
import psutil

class distributedQuery:
    indexFiles=[]
    imageDirectory=""
    currentQueryFeatureResource = None
    #Set this to true if you are testing this locally (loads the query with a single index once for faster querying)
    isTest = True
    # Set this to true if you are using socket servers to run your queries
    useServers = False
    useZip = False
    serversAndPorts = []
    defaultServerName = '127.0.0.1'
    #Directories for indexes and images
    def __init__(self, indexDir,outputImageDir,indexServerAddress=None,indexServerPort=None,tmpDir = '.',indexParamFile = None,useServer = False,det='SURF3',desc='SURF3',savefolder=None):
        self.curQuerySet = []
        self.indexFiles = []
        self.imageDirectory = outputImageDir
        make_sure_path_exists(tmpDir)
        resources = []
        if indexDir is not None and not useServer:
            for filename in os.listdir(indexDir):
                ifile = os.path.join(indexDir,filename)
                if os.path.isdir(ifile) and not self.useZip:
                    self.indexFiles.append(ifile)
                elif os.path.isfile(ifile) and self.useZip:
                    self.indexFiles.append(ifile)

            print('index files:', self.indexFiles)
        elif useServer:
            print('initializing system to connect with server at ', indexServerAddress,':',indexServerPort)
        self.curQuery = queryIndex(indexDir,indexServerAddress=indexServerAddress,indexServerPort=indexServerPort, tmpDir=tmpDir, useZip=False,indexParamFile=indexParamFile,useServer=useServer,det=det,desc=desc,savefolder=savefolder)


    def queryImages (self, queryImages, numberOfResultsToRetrieve,rootpath=''):
        allresults = []
        for i in queryImages:
            allresults.append(filteringResults())
        print('appended filteringResults')
        indCount = 0
        #print('performing query on index ', indCount, ' of ', len(self.curQuerySet))
        # This code is put here only for testing: Prevents the distributed query from reloading the index file every damn time you query an image
        # if not self.isTest:
        #     indexfile = open(index, 'rb')
        #     indexResource = Resource('index', indexfile.read(), 'application/octet-stream')
        #     print('initializing a new query index....')
        #     self.curQuery = queryIndex(indexResource)
        #print('index size: ', self.curQuery.index.ntotal)
        c=0
        #print(f'we have {len(queryImages)} images to query')
        for image in queryImages:
            print('finally almost about to query the image :O')
            result = self.curQuery.queryImage(image,numberOfResultsToRetrieve,rootpath=rootpath)
            print(result.scores)
            allresults[c].mergeScores(result)
            c=c+1
        print('iterated over query images')
        indCount += 1
        c = 0
        #print(allresults)
        return allresults

def getWorldImage(self,fileKey):
    filePath = os.path.join(self.imageDirectory,fileKey)
    worldImageResource= Resource.from_file(fileKey, filePath)
    return worldImageResource

# currently very ineffcient due to disk read. Wll parallelize
# queryImages is an array of image resourceseeryFeatures is an array of Images
def queryFeatures (self, queryFeatures, numberOfResultsToRetrieve,ignoreIDs = []):
    allresults = []
    # for i in queryImages:
    #    allresults.append(filteringResults())
    # TODO:Feature concatination for faster query batches
    # concatinate features
    # allFeats = []
    # featureExtractor = featureExtraction()
    # for feature in queryFeatures:
    #     allFeats.append(self.deserializeFeatures(feature))
    # allFeats = np.concatenate(allFeats,axis=0)
    # allFeatsResource = featureExtractor.createOutput(Resource("", featureExtractor.serializeFeature(allFeats), 'application/octet-stream'))
    # allFeatsResource = ['supplemental_information']['value']
    for i in queryFeatures:
        allresults.append(filteringResults())
    for index in self.indexFiles:
        indexfile = open(index,'rb')
        indexResource = Resource('index', indexfile.read(),'application/octet-stream')
        # curQuery = queryIndex(indexResource)
        if not self.isTest and self.curQuery is None:
            #print('initializing a new query index....')
            self.curQuery = queryIndex(indexResource)
        c=0
        for feature in queryFeatures:
            result = self.curQuery.queryFeatures(feature,numberOfResultsToRetrieve)
            allresults[c].mergeScores(result,ignoreIDs=ignoreIDs)
            c=c+1
    return allresults

def concatFeatures(self,r1,r2):
    featureExtractor = featureExtraction()
    cat = np.vstack((self.deserializeFeatures(r1['supplemental_information']['value']),self.deserializeFeatures(r2['supplemental_information']['value'])))
    filename = r1['supplemental_information']['value'].key
    featureResource = Resource(filename, featureExtractor.serializeFeature(cat), 'application/octet-stream')
    return featureExtractor.createOutput(featureResource)

if __name__ == "__main__":
    indexfolder = sys.argv[1]
    startingPort = int(sys.argv[2])
    serverNames = 'localhost'
    if len(sys.argv) > 2:
        startServers = sys.argv[3]
    startServers(serverNames,startingPort)
