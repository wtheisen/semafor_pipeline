from featureExtraction import featureExtraction

import numpy as np
import pickle
import cv2
from resources import Resource
import faiss
import indexfunctions
import fileutil
import os
import json
import collections
import socket
import time
from scoreMerge import mergeResultsMatrix
import io
import NeedleInHaystack
import math
import awkward
import scipy.stats as st
import ast
import progressbar
import gc
import dbm
from numpySocketServer import NumpySocket
from diskarray import DiskArray
from joblib import Parallel,delayed
#this class will load a single index and allow queries using either an image or features
#This allows for querying on a distributed index

class struct(object):
    pass


def loadIndexFromFile(cur,filename,id=None,tmpdir = None,no_index_load=False):
    indexfile = open(filename, 'rb')
    indexResource = Resource('index', indexfile.read(), 'application/octet-stream')
    deserializedIndexBlob = cur.deserializeIndex(indexResource,id=id,ivfStoreDir=tmpdir,no_index_load=no_index_load)
    indexfile.close()
    return deserializedIndexBlob

class queryIndex:
    runTestMode = True
    # resultSaveFolder = os.path.join('/media/jbrogan4/scratch2/Indonesia/','fdump')
    # resultSaveFolder = os.path.join('/media/wtheisen/scratch2/4chan_phash/','fdump')
    #resultSaveFolder = os.path.join('/media/wtheisen/scratch2/Indonesia_Retry','fdump')
    # resultSaveFolder = os.path.join('/media/wtheisen/scratch2/indo_vgg','fdump')
    #resultSaveFolder = os.path.join('/media/wtheisen/scratch2/indo_phash','fdump')
    #resultSaveFolder = os.path.join('/scratch365/jbrogan4/eval19/','fdump')
    #resultSaveFolder = os.path.join('/home/pthomas4/semafor/media/pthomas4/scratch2/indo_vgg', 'fdump')
    resultSaveFolder = os.path.join('/afs/crc.nd.edu/user/w/wtheisen/reddit_semafor_output', 'fdump')
    index=None
    id = None
    preproc = None
    ngpu = 0
    nprobe = 24
    tempmem = 1536 * 1024 * 1024  # if -1, use system default
    IDToImage = {}
    isDeserialized = False
    useNEHST = False
    algorithmName = "ND_dsurf_5000_filtering"
    currentQueryFeatureResource = None
    keypointMetadata = []
    featureIDToImageID = []
    imageIDToImageSize = []
    allLoadedIndexes = []
    indexFeatureOffsets = [0]
    indexImageOffsets = [0]
    rerunResults = False
    metadatadimensions = 5
    desc = 'SURF3'


    det = 'SURF3'
    no_load_index = False
    #Load Index at initialization
    #indexFileResource is a resource object
    indexServerPort = -1
    indexServerReturnPort = 9998
    indexServerAddress = 'localhost'
    def __init__(self,indexFileResource=None,indexServerAddress = None,indexServerPort=None,tmpDir=None,useZip=False,indexParamFile=None,useServer = False,det='SURF3',desc='SURF3',savefolder=None):
        print(useServer)
        if indexFileResource is not None and not useServer:
            #self.pool = Pool(processes = cpu_count)
            cacheroot = fileutil.getResourcePath(self.algorithmName)
            self.currentQueryFeatureResource = None
            self.gpu_resources = indexfunctions.wake_up_gpus(self.ngpu, self.tempmem)
            self.det = det
            self.desc = desc
            if savefolder is None:
                print('PLEASE SET THE SAVE FOLDER')
            self.resultSaveFolder = savefolder
            self.msaveFolder = os.path.join(self.resultSaveFolder,'matrices')
            self.fsaveFolder = os.path.join(self.resultSaveFolder,'features')
            print("gpus found: ", len(self.gpu_resources))
            fileutil.make_sure_path_exists(self.msaveFolder)
            fileutil.make_sure_path_exists(self.fsaveFolder)
            self.index_link_paths = []
            self.allTimes = {}
            self.allTimes['search'] = []
            self.allTimes['fusion'] = []
            self.allTimes['rank'] = []
            self.usingShards =False
            if tmpDir is None:
                tmpDir = os.path.join(os.getcwd(),'mmap_temp')
                fileutil.make_sure_path_exists(tmpDir)

            if isinstance(indexFileResource,list):
                self.init_binaryfiles(indexFileResource,tmpDir)
            elif not useZip:
                print('loading unzipped index files at ', indexFileResource)
                indexFile = os.path.join(indexFileResource,'index.index')
                dbfile = os.path.join(indexFileResource,'IDMap_db.dbm')
                metaFile = os.path.join(indexFileResource,'metadata.diskarray')
                sizeFile = os.path.join(indexFileResource ,'imagesizes.diskarray')


                self.IDToImage = dbm.open(dbfile,'c')
                if 'metaDataDim' in self.IDToImage:
                    self.metadatadimensions = int(self.IDToImage['metaDataDim'])
                    print('stored metadata dimensions: ',self.metadatadimensions)
                else:
                    print('using default meta data dimensions (',self.metadatadimensions,')')
                kmd_tmp = DiskArray(metaFile,mode='r',dtype=np.float32)
                kmd_len = int(kmd_tmp.shape[0]/self.metadatadimensions)
                kmd_tmp.close()
                kmd_tmp = DiskArray(sizeFile, mode='r', dtype=np.int)
                imsize_len = int(kmd_tmp.shape[0]/2)
                kmd_tmp.close()

                self.keypointMetadata = DiskArray(metaFile, shape=(kmd_len, self.metadatadimensions), dtype=np.float32,mode='r')
                self.imageIDToImageSize = DiskArray(sizeFile,shape=(imsize_len,2),mode='r',dtype=np.int)
                if indexParamFile is not None:
                    print('loading index parameters and preproc...')
                    indexparameters = open(indexParamFile, 'rb')
                    indexResource = Resource('indexparameters', indexparameters.read(), 'application/octet-stream')
                    indextmp, preproc, emptypathtmp, params = indexfunctions.loadIndexParameters(indexResource)
                    indexparameters.close()
                    self.preproc = preproc
                print('loading index at ', indexFile)
                self.index = faiss.read_index(indexFile)
                self.allLoadedIndexes.append(self.index)
        else:
            print('using outgoing server on port ', indexServerPort, ' and incoming server on host ', indexServerAddress, ' port ', indexServerPort)
            self.indexServerAddress = indexServerAddress
            self.indexServerPort = indexServerPort
            dbfile = os.path.join(indexFileResource, 'IDMap_db.dbm')
            self.IDToImage = dbm.open(dbfile, 'r')
            print('database length: ', len(self.IDToImage))
        self.featureExtractor = featureExtraction(detectiontype=self.det, descriptiontype=self.desc)

    def init_binaryfiles(self,indexFileResource,tmpDir=None):
        if isinstance(indexFileResource,list):
            print('loading ', len(indexFileResource), ' multiple index files...')
            #self.allLoadedIndexes = Parallel(n_jobs=max(5,len(indexFileResource)))(delayed(loadIndexFromFile)(self,filename) for filename in indexFileResource[1:])
            self.imageIDToImageSize = None
            self.imageIDtoSizeDict = {}
            self.featureIDToImageID = None
            self.keypointMetadata = None
            self.index_link_paths = []
            self.allTimes = {}
            self.allTimes['search'] = []
            self.allTimes['fusion'] = []
            self.allTimes['disk'] = []
            self.allTimes['rank'] =[]
            self.resultSaveFolder = os.path.join(tmpDir,'fdump')
            fileutil.make_sure_path_exists(self.resultSaveFolder)
            #indexFileResource = indexFileResource[:1]
            count = 0
            indLengths = []
            metadataWidth = -1
            if not self.rerunResults:
                bar = progressbar.ProgressBar()
                self.no_load_index = False
                print('building mmap files... ')
                for filename in bar(indexFileResource):
                    indexfile = open(filename, 'rb')
                    byteranges = fileutil.getMultiFileByteArraySizes_fast(indexfile,12,8)
                    indexfile.seek(0)
                    indexfile.seek(byteranges[4][0]+10)
                    indexDesc = ast.literal_eval(indexfile.read(118).decode('ascii').split('}')[0]+'}')
                    #print(indexDesc['shape'])
                    indShape = indexDesc['shape']
                    if len(indShape) > 1 and  indShape[1] > metadataWidth:
                        metadataWidth = indShape[1]
                    else:
                        metadataWidth = 6
                    indLengths.append(indShape[0])
                    indexfile.close()
                totalIndLength = sum(indLengths)
                #print('indLengths ',indLengths)
                self.keypointMetadata = np.memmap(os.path.join(tmpDir,'tmp_kmd.npy'), dtype='float32', mode='w+', shape=(totalIndLength, metadataWidth))
                #print('md shape: ',self.keypointMetadata.shape)
                self.imageIDToImageSize = None #np.memmap(os.path.join(tmpDir,'tmp_idtosize'), dtype='float32', mode='w+', shape=(totalIndLength, 2))
                self.featureIDToImageID = np.memmap(os.path.join(tmpDir,'tmp_feattoim'), dtype='float32', mode='w+', shape=(totalIndLength))
                bar = progressbar.ProgressBar()
                #tmpdir = '/scratch365/jbrogan4/eval19/tmp'
                tmpdir = tmpDir#'/media/jbrogan4/scratch2/eval19/tmp'
                for filename in bar(indexFileResource):
                    res = loadIndexFromFile(self,filename,id=count,tmpdir = tmpdir,no_index_load=self.no_load_index)
                    idToImageName = res[2]
                    imageIDtoSize = res[5]
                    self.index_link_paths.append(res[6])
                    #Append image ID to image name dict
                    imgIDKeys = list(map(str,list(np.array(np.array(list(map(int, idToImageName.keys()))) + self.indexImageOffsets[-1],dtype=np.int))))
                    self.IDToImage.update(dict(zip(imgIDKeys, idToImageName.values())))
                    del imgIDKeys
                    #Append image ID to image size dict
                    imIDKeys = list(np.array(np.array(list(map(int, imageIDtoSize.keys()))) + self.indexImageOffsets[-1],dtype=np.int))
                    self.imageIDtoSizeDict.update(dict(zip(imIDKeys,imageIDtoSize.values())))
                    #imIDKeysSorted = imIDKeys.argsort()
                    idsizearr = np.zeros((len(imageIDtoSize),2))
                    for i in imageIDtoSize:
                        idsizearr[int(i)] = imageIDtoSize[i]
                    #idsizearr = np.array(list(imageIDtoSize.values()))[imIDKeysSorted]
                    if self.imageIDToImageSize is None:
                        self.imageIDToImageSize = idsizearr
                    else:
                        self.imageIDToImageSize = np.concatenate((self.imageIDToImageSize,idsizearr))
                    #self.imageIDToImageSize[int(self.indexImageOffsets[-1]):int(self.indexImageOffsets[-1]+len(imIDKeysSorted))] = np.array(list(imageIDtoSize.values()))[imIDKeysSorted]
                    #del imIDKeysSorted
                    del imageIDtoSize
                    #Append main index
                    self.allLoadedIndexes.append(res[0])
                    #Append feature ID to image ID table

                    self.featureIDToImageID[self.indexFeatureOffsets[-1]:self.indexFeatureOffsets[-1]+len(res[4])] = res[4] + self.indexImageOffsets[-1]
                    #Append how many features are in this index
                    self.indexFeatureOffsets.append(len(res[4])+self.indexFeatureOffsets[-1])
                    #Append how many images are in this index
                    self.indexImageOffsets.append(res[4].max()+1+self.indexImageOffsets[-1])
                    #Append the keypoint metadata
                    self.keypointMetadata[self.indexFeatureOffsets[-2]:self.indexFeatureOffsets[-2]+res[3].shape[0],:] = res[3]
                    self.preproc = res[1]
                    #print('memory usage: ', psutil.virtual_memory().percent)
                    count += 1
                    gc.collect()
                    #print('d_in for preproc: ',self.preproc.d_in)
            #add final overflow index to id table and metadata table
            self.featureIDToImageID[-1] = -1
            self.keypointMetadata[-1,:] = [-1,-1,-1,-1,-1,-1]


        else:
            self.deserializeIndex(indexFileResource,id)


    def queryImage_batch(self,imageResources,numberOfResultsToRetrieve,cores=1):
        allFeatures = []
        allResults = []
        #print('extracting features from batch')
        if cores == 1:
            bar = progressbar.ProgressBar()
            for imageResource in bar(imageResources):
                features,features_r = extractImageFeatures(imageResource,self.featureExtractor)
                features_all = self.concatFeatures(features,features_r)
                allFeatures.append(features_all)
        else:
            allFeatures = Parallel(n_jobs=cores)(delayed(extractImageFeatures)(imageResource,self.featureExtractor) for imageResource in imageResources)
        #print('querying features')
        bar = progressbar.ProgressBar()
        for features_all in bar(allFeatures):
            allResults.append(self.queryFeatures(features_all['supplemental_information']['value'], numberOfResultsToRetrieve))


    #queryImage conatins resourse object containing image
    def queryImage (self, imageResource, numberOfResultsToRetrieve,rootpath=''):
        #create score object
        #get probe features
        if not self.rerunResults:
            feature,feature_r = extractImageFeatures(imageResource,self.featureExtractor,useFlip=False,rootpath=rootpath)

            #print('extracted image features?')
            if feature_r is None:
                features_all = feature
            else:
                features_all = self.concatFeatures(feature, feature_r)
            if features_all is not None:
                #if not os.path.exists(savedFeatPath) and False:
                #   with open(os.path.join(self.fsaveFolder,imageResource.key), 'wb') as of:
                #      of.write(featureExtraction.serializeFeature(features_all['supplemental_information']['value']._data,))
                results = self.queryFeatures(features_all['supplemental_information']['value'], numberOfResultsToRetrieve)
        else:
            results = self.queryFeatures()

        return results

    def performQueryLocal(self,features,numberOfResultsToRetrieve):
        tstart = time.time()
        query_feature_metadata = features[:, -4:]
        features = features[:,:-4]
        print('FOTES: ', features.shape)
        quert = indexfunctions.sanitize(features)
        print('sanitized')
        pfeatures = self.preproc.apply_py(quert)
        print("preproc'ed")
        usemmap = False
        if usemmap:
            allI = np.memmap('fusetmpI.tmp', mode='w+',
                             shape=(pfeatures.shape[0], numberOfResultsToRetrieve * len(self.allLoadedIndexes)),
                             dtype=np.int)
            allD = np.memmap('fusetmpD.tmp', mode='w+',
                             shape=(pfeatures.shape[0], numberOfResultsToRetrieve * len(self.allLoadedIndexes)),
                             dtype=np.float32)
        else:
            allI = np.zeros((pfeatures.shape[0], numberOfResultsToRetrieve * len(self.allLoadedIndexes)), dtype=np.int)
            allD = np.zeros((pfeatures.shape[0], numberOfResultsToRetrieve * len(self.allLoadedIndexes)),
                            dtype=np.float32)
        colOffset = 0
        startSearchTime = time.time()
        print('starting search')
        resCount = 0
        for index in self.allLoadedIndexes:
            t0 = time.time()
            Dt, It = index.search(pfeatures, numberOfResultsToRetrieve)
            It = It + self.indexFeatureOffsets[resCount]
            It[It >= len(self.keypointMetadata) - 1] = len(self.keypointMetadata) - 1
            It[It < 0] = len(self.keypointMetadata) - 1
            allI[:, colOffset:colOffset + It.shape[1]] = It.astype(int)
            allD[:, colOffset:colOffset + It.shape[1]] = Dt.astype(np.float32)
            colOffset += It.shape[1]
            resCount += 1
            t1 = time.time()
            #print('query took ', t1 - t0, ' seconds')
            gc.collect()
        endSearchTime = time.time()
        #print('total search time: ', endSearchTime - startSearchTime, ' seconds')
        self.allTimes['search'].append(endSearchTime - startSearchTime)
        # Fuse returned results from all indexes lossleslly
        print('starting fusion..')
        tstart1 = time.time()
        # allI = np.concatenate(allI, axis=1)
        # allD = np.concatenate(allD,axis=1)

        # allMeta = np.concatenate(allMeta,axis=1)
        # belongsToIndex = np.concatenate(belongsToIndex,axis=1)
        sortedArgs = allD.argsort(axis=1)
        inds = (sortedArgs + (np.arange(sortedArgs.shape[0]) * sortedArgs.shape[1])[:, np.newaxis]).flatten()

        D = allD.flatten()[inds].reshape(allD.shape)  # [:,:numberOfResultsToRetrieve]
        I = allI.flatten()[inds].reshape(allI.shape)  # [:,:numberOfResultsToRetrieve]
        tend1 = time.time()
        #print('fusion time: ', tend1 - tstart1, ' seconds')
        self.allTimes['fusion'].append(tend1 - tstart1)
        # np.save(Ipath,I)
        # np.save(dpath,D)
        Iflat = I.flatten()
        # image_level_IDs = self.featureIDToImageID[I.flatten()].reshape(I.shape).astype(np.int)
        disktime = time.time()
        sortedI_inds = Iflat.argsort()
        I_originalOrder = np.arange(len(Iflat))[sortedI_inds]
        #print(f'I: {I}')
        Isorted = Iflat[sortedI_inds]
        print('meow')
        #print(f'isorted: {Isorted}')
        allmd_S = self.keypointMetadata.data[Isorted]
        print('cat')
        allmd = np.zeros_like(allmd_S)
        allmd[I_originalOrder] = allmd_S
        del allmd_S
        image_level_IDs = allmd[:, -1].astype(np.int).reshape((I.shape[0], I.shape[1]))
        imagesizes = self.imageIDToImageSize[image_level_IDs.flatten()]
        allMetaDatabase = allmd.reshape(
            (I.shape[0], I.shape[1], self.keypointMetadata.shape[1]))
        disktime2 = time.time()
        allMetaQuery = query_feature_metadata.reshape(
            (query_feature_metadata.shape[0], 1, query_feature_metadata.shape[1])).repeat(I.shape[1],
                                                                                          axis=1)
        print('Not saving it?')
        saveit= False
        if saveit:
            fileutil.make_sure_path_exists(os.path.dirname(self.dpath))
            np.save(self.dpath,D)
            np.save(self.Ipath,I)
            np.save(self.dbmetapath,allMetaDatabase)
            np.save(self.qmetapath,allMetaQuery)
            np.save(self.ilidpath,image_level_IDs)
            np.save(self.sizespath,imagesizes)
        #print('disk time: ', disktime2 - disktime, ' seconds')
        return I,D,image_level_IDs,imagesizes,allMetaDatabase,allMetaQuery

    def performQueryServer(self,features):
        #features = features[:, :-4]
        npSocket = NumpySocket()
        npSocket.startServer(self.indexServerAddress, int(self.indexServerPort))
        npSocket.sendNumpy(features)
        npSocket.socket.close()
        #Start listening for returned data from server
        npSocket2 = NumpySocket()
        print('starting client')
        npSocket2.startClient(int(self.indexServerPort)-1)
        time.sleep(.5)
        retrievedData = npSocket2.recieveNumpy()
        npSocket2.socket.close()
        print('recieved data back')
        return retrievedData[0],retrievedData[1],retrievedData[2],retrievedData[3],retrievedData[4],retrievedData[5]

    #queryFeature contains resource object containing feature(s)
    #this allows for non-image queries
    def queryFeatures (self, featureResource, numberOfResultsToRetrieve,resourceName = None):
        useNehst = False
        #useNehst = True
        if resourceName is None and not self.rerunResults:
            resourceName = featureResource.key
        unqpath = os.path.join(self.msaveFolder,resourceName)
        self.dpath = os.path.join(unqpath, 'D.npy')
        self.Ipath = os.path.join(unqpath ,'I.npy')
        self.dbmetapath = os.path.join(unqpath,'metadb.npy')
        self.qmetapath = os.path.join(unqpath, 'metaq.npy')
        self.ilidpath = os.path.join(unqpath, 'image_level_IDs.npy')
        self.sizespath = os.path.join(unqpath, 'imageSizes.npy')
        resultScores_empty = filteringResults()
        resultScores_empty.addScore('empty', 1, ID=0)
        resultScores_empty.addScore('t2', .5, ID=1)
        gc.collect()
        numberOfResultsToRetrieve = int(numberOfResultsToRetrieve)

        ps = None

        if self.ngpu > 0:
            ps = faiss.GpuParameterSpace()
            ps.initialize(self.index)
            #ps.set_index_parameter(self.index, 'nprobe', self.nprobe)
        features,imgshape = self.deserializeFeatures(featureResource)
        if len(self.allLoadedIndexes) > 0:
            I, D, image_level_IDs, imagesizes, allMetaDatabase, allMetaQuery=self.performQueryLocal(features,numberOfResultsToRetrieve)
        else:
            I, D, image_level_IDs, imagesizes, allMetaDatabase, allMetaQuery=self.performQueryServer(features)
        if not os.path.exists(unqpath) and not self.no_load_index:
            os.makedirs(unqpath)

        tallyStart = time.time()
        if useNehst:
            sortedIDs, sortedVotes,maxvoteval = NeedleInHaystack.nhscore(I,D,image_level_IDs,imagesizes,allMetaDatabase,allMetaQuery,visualize=False,numberOfResultsToRetrieve=numberOfResultsToRetrieve)
        else:
            print('using Tally scoring instead of NH scoring')
            sortedIDs, sortedVotes, maxvoteval = indexfunctions.tallyVotes(D, I, image_level_IDs, numcores=1)
            print('max scores: ', sortedVotes[:10])
        tallyEnd = time.time()

        print('ranking time: ',tallyEnd-tallyStart,' seconds')
        #self.allTimes['rank'].append(tallyEnd - tallyStart)
        timeavs = {}
        #for c in self.allTimes:
        #    timeavs[c] = np.array(self.allTimes[c]).mean()
        #print('average times:\n',timeavs)

        resultScores = filteringResults()
        for i in range(0, min(len(sortedIDs), numberOfResultsToRetrieve*10)):
            id = sortedIDs[i]
            id_str = str(int(id))
            #print(id_str)
            if id_str in self.IDToImage:
                imname = self.IDToImage[id_str].decode('ascii')
                score = sortedVotes[i]
                resultScores.addScore(imname,score,ID=id)
        resultScores.pairDownResults(numberOfResultsToRetrieve)
        return resultScores
    # @profile(precision=5)

    def deserializeIndex (self, indexFileResource,id=None,ivfStoreDir = './',useCache=True,no_index_load=False):
        # the joined index file contains populated_index,empty_trained_index,preproc,IDMap, (possible) IVF file, in that order
        bytearrays = fileutil.splitMultiFileByteArray(indexFileResource._data,12,8)
        tmppath = os.path.join(ivfStoreDir,'tmp')
        try:
            os.makedirs(tmppath)
        except:
            pass
        mv = memoryview(indexFileResource._data)
        if id is not None:
            tmppath += '_'+str(id)
        all_tmp_paths = []
        mmap_path = None
        count = 0
        if ivfStoreDir is not None:
            try:
                os.makedirs(ivfStoreDir)
            except:
                pass

        for bytearray in bytearrays:
            p = tmppath + str(count) + '.dat'
            if os.path.exists(p) and not count == 7 and not count == 6:
                count += 1
                all_tmp_paths.append(p)
                continue
            if count == 6:
                # This is the path to where the ivf mmap file must be stored
                mmap_path = mv[bytearray[0]:bytearray[1]].tobytes().decode('ascii')
                mmap_name = os.path.basename(mmap_path)
                print('mmap_path: ',mmap_path)

                realSavePath = os.path.join(ivfStoreDir,mmap_name+str(id).zfill(3))
                print('realsavepath: ', realSavePath)
                #print('making directory ', os.path.dirname(mmap_path), ' to store ivf data')
                if not os.path.exists(os.path.dirname(mmap_path)):
                    try:
                        os.makedirs(os.path.dirname(mmap_path))
                    except:
                        pass

            if count == 7:
                #p = mmap_path
                p = realSavePath
                print('path needed to acess on-disk ivf:', realSavePath)
            if count == 7 and no_index_load:
                count+=1
                continue
            if os.path.exists(p):
                count += 1
                all_tmp_paths.append(p)
                continue
            with open(p,'wb') as fp:
                fp.write(mv[bytearray[0]:bytearray[1]])
            count+=1
            all_tmp_paths.append(p)
        print('preproc path: ',all_tmp_paths[2])
        preproc = faiss.read_VectorTransform(all_tmp_paths[2])
        print('loaded preproc d_in: ',preproc.d_in)
        IDToImage = json.loads(mv[bytearrays[3][0]:bytearrays[3][1]].tobytes().decode('ascii'))
        keypointMetadata = np.load(io.BytesIO(mv[bytearrays[4][0]:bytearrays[4][1]].tobytes()))
        featureIDToImageID = keypointMetadata[:,4]
        imageIDToImageSize = json.loads(mv[bytearrays[5][0]:bytearrays[5][1]].tobytes().decode('ascii'))
        #print(mv[bytearrays[3][0]:bytearrays[3][1]].tobytes())
        del bytearrays
        del indexFileResource
        #print('initializing index...')
        index = None
        if not no_index_load: #not self.runTestMode:
            #mmap_path='/scratch365/jbrogan4/Medifor/GPU_Prov_Filtering/provenance/ND_surf_low_5000_filtering/algorithmResourcesmerged_index__.ivfdata'
            print('sylinking index:', realSavePath, ' --> ', mmap_path)

            os.symlink(realSavePath,mmap_path)
            index = faiss.read_index(all_tmp_paths[0],faiss.IO_FLAG_MMAP)
            os.remove(mmap_path)
            #print('index size: ',index.ntotal)
            #print('map size:',len(IDToImage))
        isDeserialized = True
        return(index,preproc,IDToImage,keypointMetadata,featureIDToImageID,imageIDToImageSize,(realSavePath,mmap_path))

    def deserializeFeatures(self, featureResource):
        data = featureResource._data
        return np.reshape(data[:-4], (int(data[-4]), int(data[-3]))),(data[-2],data[-1])

    def concatFeatures(self,r1,r2):
        featureExtractor = featureExtraction(descriptiontype='DELF',detectiontype='DELF')
        f1 = self.deserializeFeatures(r1['supplemental_information']['value'])
        f2 = self.deserializeFeatures(r2['supplemental_information']['value'])
        cat = np.vstack((f1[0],f2[0]))
        filename = r1['supplemental_information']['value'].key
        featureResource = Resource(filename, featureExtractor.serializeFeature(cat), 'application/octet-stream')
        return featureExtractor.createOutput(featureResource)


#Thiss class# produces the data needed for the Provenance Filtering JSON
#the function merge will be used to merge results when indexing is parallelized
# you can modify the class implementations to meet your needs, but function calls
# should be kept the same
class filteringResults:
    map = {}
    scores = collections.OrderedDict()
    def __init__(self):
        self.probeImage = ""
        self.I = None
        self.D = None
        self.map = {}
        self.scores = collections.OrderedDict()
        self.visData = collections.OrderedDict()
    def addScore(self,filename, score,ID=None,visData=None):
        self.scores[filename]=score
        if ID is not None:
            self.map[ID] = filename
        if visData is not None:
            self.visData[filename] = visData
    #this function merges two results
    def mergeScores(self,additionalScores,ignoreIDs = []):
        if self.I is not None and self.D is not None and additionalScores is not None and additionalScores.I is not None and additionalScores.D is not None:
            print('merging scores using I and D')
            # Merge results based on I and D matrixes (not heuristic!)
            mergedresults = mergeResultsMatrix(self.D,additionalScores.D,self.I,additionalScores.I,self.map,additionalScores.map,k=min(len(self.scores),self.I.shape[1]),numcores=12)
            self.I = mergedresults[0]
            self.D = mergedresults[1]
            self.map = mergedresults[2]
            sortedIDs, sortedVotes,maxvoteval = indexfunctions.tallyVotes(self.D, self.I)
            # voteScores = 1.0 * sortedVotes / (1.0 * np.max(sortedVotes))
            voteScores = 1.0 * sortedVotes / (maxvoteval)
            self.scores = collections.OrderedDict()
            for i in range(0, len(sortedIDs)):
                id = sortedIDs[i]
                if id not in ignoreIDs:
                    id_str = str(id)
                    if id in self.map:
                        imname = self.map[id]
                        score = voteScores[i]
                        self.addScore(imname, score, ID=id)

        elif additionalScores is None:
            #if additional scores contains nothing don't add anything!
            pass
        elif self.I is None and self.D is None and additionalScores.I is not None and additionalScores.D is not None:
            # Pushing into empty results, just populate the object with the additionalScores
            self.I = additionalScores.I
            self.D = additionalScores.D
            self.map = additionalScores.map
            self.scores = additionalScores.scores

        else:
            # Merge in a heuristic way
            self.scores.update(additionalScores.scores)
            if additionalScores.visData is not None:
                self.visData.update(additionalScores.visData)
                #sortinds = np.array(self.scores.values()).argsort()
                #vd = self.visData.copy()
                #self.visData.clear()
                #for v in np.array(list(vd.keys())).argsort()[::-1]:
                #   self.visData[v] = vd[v]
            sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
            self.scores = sortedscores
        for id in ignoreIDs:
            if id in self.scores:
                del self.scores[id]
    # this function merges two results
    def dictSort(self, additionalScores):
        od = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores.update(additionalScores.scores)
        sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores = sortedscores

    #Once scores are merged together, at most "numberOfResultsToRetrieve" will be retained
    def pairDownResults(self,numberOfResultsToRetrieve):
        numberOfResultsToRetrieve = int(numberOfResultsToRetrieve)
        if len(self.scores) > numberOfResultsToRetrieve:
            newscores = collections.OrderedDict(
                sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:numberOfResultsToRetrieve])
            self.scores = newscores
    def normalizeResults(self):
        maxVal = self.scores[list(self.scores.keys())[0]]
        for s in self.scores:
            self.scores[s] = self.scores[s]/maxVal
def extractImageFeatures(imageResource,featureExtractor,useFlip=True,rootpath=''):
    features_all = None
    fpath = os.path.join(rootpath,imageResource.key)
    print('resource: ',fpath)
    feature = featureExtractor.processImage(imageResource, tfcores=24,resourcepath= fpath)
    feature_r = None
    if feature is not None and useFlip:
        feature_r = featureExtractor.processImage(imageResource, flip=True, tfcores=24,resourcepath=fpath)
    return feature,feature_r

if __name__ == "__main__":
    im1 ='/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/probe/ff7c1f46c84e6efce1cb563bf9d1b65d.JPG'
    im2 = '/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/probe/797a0b5dd9c47a4b391f4cee60ba9354.jpg'
    indexFile = '/home/jbrogan4/Documents/Projects/Medifor/GPU_Prov_Filtering/provenance/tutorial/index/index'
    r = Resource.from_file(os.path.basename(im1),im1)
    indexfile = open(indexFile, 'rb')
    indexResource = Resource('index', indexfile.read(), 'application/octet-stream')

    #
    print('here')
