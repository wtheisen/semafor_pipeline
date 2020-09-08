import os
import sys
import faiss
import cv2
import fileutil
from diskarray import DiskArray
import numpy as np
import indexfunctions
from resources import Resource
from numpySocketServer import NumpySocket
import time
import dbm
import gc

class indexServer:
    runTestMode = True
    resultSaveFolder = os.path.join('/media/jbrogan4/scratch2/eval19/','fdump')
    #resultSaveFolder = os.path.join('/scratch365/jbrogan4/eval19/','fdump')
    msaveFolder = os.path.join(resultSaveFolder,'matrices')
    fsaveFolder = os.path.join(resultSaveFolder,'features')
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
    metadatadimensions = 5

    no_load_index = False
    #Load Index at initialization
    #indexFileResource is a resource object

    def __init__(self,indexFileResource,tmpDir=None,indexParamFile=None,):
        cacheroot = fileutil.getResourcePath(self.algorithmName)
        self.currentQueryFeatureResource = None

        self.gpu_resources = indexfunctions.wake_up_gpus(self.ngpu, self.tempmem)
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
        print('loading unzipped index files at ', indexFileResource)
        indexFile = os.path.join(indexFileResource,'index.index')
        dbfile = os.path.join(indexFileResource,'IDMap_db.dbm')
        metaFile = os.path.join(indexFileResource,'metadata.diskarray')
        sizeFile = os.path.join(indexFileResource ,'imagesizes.diskarray')


        self.IDToImage = dbm.open(dbfile,'r')
        if 'metaDataDim' in self.IDToImage:
            self.metadatadimensions = self.IDToImage['metaDataDim']
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
    def listenLoop(self,numberOfResultsToRetrieve):


        npSocket2 = None
        while (True):
            try:
                npSocket = NumpySocket()
                npSocket.startClient(9999)
                # Capture frame-by-frame
                print('waiting for image features...')
                features = npSocket.recieveNumpy()
                npSocket.socket.close()
                print('recieved features!')


                query_feature_metadata = features[:, -4:]
                features = features[:, :-4]
                print('featuredim ',features.shape)
                pfeatures = self.preproc.apply_py(indexfunctions.sanitize(features))
                usemmap = False
                if usemmap:
                    allI = np.memmap('fusetmpI.tmp', mode='w+',
                                     shape=(pfeatures.shape[0], numberOfResultsToRetrieve * len(self.allLoadedIndexes)),
                                     dtype=np.int)
                    allD = np.memmap('fusetmpD.tmp', mode='w+',
                                     shape=(pfeatures.shape[0], numberOfResultsToRetrieve * len(self.allLoadedIndexes)),
                                     dtype=np.float32)
                else:
                    allI = np.zeros((pfeatures.shape[0], numberOfResultsToRetrieve * len(self.allLoadedIndexes)),
                                    dtype=np.int)
                    allD = np.zeros((pfeatures.shape[0], numberOfResultsToRetrieve * len(self.allLoadedIndexes)),
                                    dtype=np.float32)
                colOffset = 0
                startSearchTime = time.time()
                resCount = 0
                print('starting search')
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
                    print('query took ', t1 - t0, ' seconds')
                    gc.collect()
                endSearchTime = time.time()
                print('total search time: ', endSearchTime - startSearchTime, ' seconds')
                self.allTimes['search'].append(endSearchTime - startSearchTime)
                # Fuse returned results from all indexes lossleslly
                print('starting fusion..')
                tstart1 = time.time()
                sortedArgs = allD.argsort(axis=1)
                inds = (sortedArgs + (np.arange(sortedArgs.shape[0]) * sortedArgs.shape[1])[:, np.newaxis]).flatten()

                D = allD.flatten()[inds].reshape(allD.shape)  # [:,:numberOfResultsToRetrieve]
                I = allI.flatten()[inds].reshape(allI.shape)  # [:,:numberOfResultsToRetrieve]
                tend1 = time.time()
                print('fusion time: ', tend1 - tstart1, ' seconds')
                self.allTimes['fusion'].append(tend1 - tstart1)
                Iflat = I.flatten()
                disktime = time.time()
                sortedI_inds = Iflat.argsort()
                I_originalOrder = np.arange(len(Iflat))[sortedI_inds]
                Isorted = Iflat[sortedI_inds]
                allmd_S = self.keypointMetadata.data[Isorted]
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
                print('disk time: ', disktime2 - disktime, ' seconds')
                tallyStart = time.time()
                sendback = np.array([I,D,image_level_IDs,imagesizes,allMetaDatabase,allMetaQuery])
            except:
                print('error retrieving results')
                sendback = np.array([[],[],[],[],[],[]])
            print('starting server')
            npSocket2 = NumpySocket()
            for i in range(100):
                isconnected = npSocket2.startServer('localhost', 9998)
                if isconnected == 1:
                    break
                else:
                    print('try to connect socket 2 again')
            print('sending retrieval data back...')
            sendtime0 = time.time()
            npSocket2.sendNumpy(sendback)
            npSocket2.socket.close()
            sendtime1 = time.time()
            print('sent in ', sendtime1-sendtime0,' seconds')

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    server = indexServer('/media/jbrogan4/scratch2/Indonesia/index/instagram_small',None,indexParamFile='/media/jbrogan4/scratch2/Indonesia/indextraining/parameters_twitter_small')
    server.listenLoop(500)