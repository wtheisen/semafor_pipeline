from featureExtraction import featureExtraction

import numpy as np
from resources import Resource
import faiss
import indexfunctions
import fileutil
import os
import collections
import time
from scoreMerge import mergeResultsMatrix
import NeedleInHaystack
import gc
import dbm
from diskarray import DiskArray

class queryIndex:
    runTestMode = True
    index=None
    id = None
    preproc = None
    ngpu = 1
    nprobe = 24
    # tempmem = 1536 * 1024 * 1024  # if -1, use system default
    tempmem = -1
    IDToImage = {}
    isDeserialized = False
    useNEHST = True
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

    def __init__(self,indexFileResource=None,tmpDir=None,indexParamFile=None,det='SURF3',desc='SURF3',savefolder=None):
        cacheroot = fileutil.getResourcePath(self.algorithmName)
        self.currentQueryFeatureResource = None
        self.gpu_resources = indexfunctions.wake_up_gpus(self.ngpu, self.tempmem)
        self.det = det
        self.desc = desc
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

        tmpDir = os.path.join(os.getcwd(),'mmap_temp')
        fileutil.make_sure_path_exists(tmpDir)

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

        self.featureExtractor = featureExtraction(detectiontype=self.det, descriptiontype=self.desc)

    #queryImage conatins resourse object containing image
    def queryImage (self, imageResource, recall,rootpath=''):
        if not self.rerunResults:
            feature, feature_r = extractImageFeatures(imageResource,self.featureExtractor,useFlip=False,rootpath=rootpath)

            if feature_r is None:
                features_all = feature
            else:
                features_all = self.concatFeatures(feature, feature_r)
            if features_all is not None:
                results = self.queryFeatures(features_all['supplemental_information']['value'], recall)
        else:
            results = self.queryFeatures()

        return results

    def performQueryLocal(self, features, recall):
        tstart = time.time()
        query_feature_metadata = features[:, -4:]
        features = features[:,:-4]
        print('FOTES: ', features.shape)
        quert = indexfunctions.sanitize(features)
        print('sanitized')
        pfeatures = self.preproc.apply_py(quert)
        print("preproc'ed")
        allI = np.zeros((pfeatures.shape[0], recall * len(self.allLoadedIndexes)), dtype=np.int)
        allD = np.zeros((pfeatures.shape[0], recall * len(self.allLoadedIndexes)),
                        dtype=np.float32)
        colOffset = 0
        startSearchTime = time.time()
        print('starting search')
        resCount = 0

        for index in self.allLoadedIndexes:
            t0 = time.time()
            Dt, It = index.search(pfeatures, recall)
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
        self.allTimes['search'].append(endSearchTime - startSearchTime)
        # Fuse returned results from all indexes lossleslly
        tstart1 = time.time()
        sortedArgs = allD.argsort(axis=1)
        inds = (sortedArgs + (np.arange(sortedArgs.shape[0]) * sortedArgs.shape[1])[:, np.newaxis]).flatten()

        D = allD.flatten()[inds].reshape(allD.shape)  # [:,:recall]
        I = allI.flatten()[inds].reshape(allI.shape)  # [:,:recall]
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
        return I,D,image_level_IDs,imagesizes,allMetaDatabase,allMetaQuery

    #queryFeature contains resource object containing feature(s)
    #this allows for non-image queries
    def queryFeatures (self, featureResource, recall,resourceName = None):
        #useNehst = False
        useNehst = True
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
        recall = int(recall)

        ps = None

        if self.ngpu > 0:
            ps = faiss.GpuParameterSpace()
            ps.initialize(self.index)
            #ps.set_index_parameter(self.index, 'nprobe', self.nprobe)
        features,imgshape = self.deserializeFeatures(featureResource)
        I, D, image_level_IDs, imagesizes, allMetaDatabase, allMetaQuery=self.performQueryLocal(features,recall)

        if not os.path.exists(unqpath) and not self.no_load_index:
            os.makedirs(unqpath)

        tallyStart = time.time()
        if useNehst:
            sortedIDs, sortedVotes,maxvoteval = NeedleInHaystack.nhscore(I,D,image_level_IDs,imagesizes,allMetaDatabase,allMetaQuery,visualize=False,recall=recall)
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
        for i in range(0, min(len(sortedIDs), recall*10)):
            id = sortedIDs[i]
            id_str = str(int(id))
            #print(id_str)
            if id_str in self.IDToImage:
                imname = self.IDToImage[id_str].decode('ascii')
                score = sortedVotes[i]
                resultScores.addScore(imname,score,ID=id)
        resultScores.pairDownResults(recall)
        return resultScores

    def deserializeFeatures(self, featureResource):
        data = featureResource._data
        return np.reshape(data[:-4], (int(data[-4]), int(data[-3]))),(data[-2],data[-1])

    def concatFeatures(self,r1,r2):
        featureExtractor = featureExtraction(descriptiontype='SURF3',detectiontype='SURF3')
        f1 = self.deserializeFeatures(r1['supplemental_information']['value'])
        f2 = self.deserializeFeatures(r2['supplemental_information']['value'])
        cat = np.vstack((f1[0],f2[0]))
        filename = r1['supplemental_information']['value'].key
        featureResource = Resource(filename, featureExtractor.serializeFeature(cat), 'application/octet-stream')
        return featureExtractor.createOutput(featureResource)

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

    #Once scores are merged together, at most "recall" will be retained
    def pairDownResults(self,recall):
        recall = int(recall)
        if len(self.scores) > recall:
            newscores = collections.OrderedDict(
                sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:recall])
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
