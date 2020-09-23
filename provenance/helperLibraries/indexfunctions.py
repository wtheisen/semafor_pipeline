print('started indexfunctions import')
import os
import sys
import numpy as np
# import cv2
import faiss
print('starting joblib')
from joblib import Parallel,delayed
from multiprocessing import Manager,Process
import progressbar
import fileutil

print('starting unravel index')

unravel_index = np.unravel_index
npsum = np.sum
prog_q = Manager().Queue(1)
quitProg = Manager().Value('l', True)
print('starting SCSM')
import SCSM
print(SCSM.__file__)
print("ended indexfunctions import")

def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

def weightVotes(locs,shape):
    locCoords = unravel_index(locs,shape)
    return npsum(1/(locCoords[1]+1))

def progressQueueThread(length):
    b = progressbar.ProgressBar(max_value=length)
    i=0
    while quitProg.value:
        j = prog_q.get(timeout=30)
        b.update(i)
        i+=1

def startMultiThreadedProgress(length):
    prog_q = Manager().Queue(length)
    p1 = Process(target=progressQueueThread, args=(length,), )

def adaptiveRerank(D,rank): #Adaptive feature ranking based on "Exploiting descriptor distances for precise image search" page 7
    threshD = D.copy()
    badInds = threshD > 100
    threshD[badInds] = 0
    lowestRanks = np.argmax(threshD,axis=1)
    # maxD = threshD.max()
    adaptiveFeatureScores = D[(range(0,D.shape[0]), lowestRanks)].reshape((threshD.shape[0],1))-D
    adaptiveFeatureScores[adaptiveFeatureScores < 0] = 0
    return adaptiveFeatureScores
def inverseRerank(D):
    invD = 1/(1+D) #constrains our score between 0 and 1
    return invD
def single_scsm_vote(probeim,input): #(imgID,id_loc,imgSize,query_kp,keypoint_kp)
    return SCSM.SCSM_kp(input[0],input[1],input[2],input[3],input[4],probeim,input[5])

def scsmVotes_vectorized(D,imageLevel_I,query_keypoint_metadata,database_keypoint_metadata_tensor,imgIDtoShapeDict):
    pass

def scsmVotes(D, featureLevel_I,featureIDtoImageID,query_keypoint_metadata,database_keypoint_metadata,imgIDtoShapeDict,jobNum=1,IDtoImage=None,qimgPath=None,imageLevelI=None):
    distThresh=10
    if imageLevelI is None:
        I = featureIDtoImageID[featureLevel_I.flatten()]
    else:
        I = imageLevelI
    Ishape = featureLevel_I.shape
    I = np.reshape(I, featureLevel_I.shape)
    I[D > distThresh] = -1
    D2 = D[D < distThresh]
    avgDist = D2.mean()
    distSTD = D2.std()
    ids, unq_inv, votes = np.unique(I, return_inverse=True, return_counts=True)
    ids = ids.astype(np.int)
    unq_inv_s = np.argsort(unq_inv)
    id_locs = np.split(unq_inv_s, np.cumsum(votes[:-1]))
    #adaptiveScores = adaptiveRerank(D,D.shape[1]-1)
    adaptiveScores = inverseRerank(D)
    scores_for_centroids = []
    ids_for_centroids = []
    bar = progressbar.ProgressBar()
    i = 0
    id_loc_lengths = []
    for l in id_locs:
        id_loc_lengths.append(len(l))
    id_loc_lengths = np.asarray(id_loc_lengths)
    id_locs_sorted_by_amount_indexes = np.argsort(id_loc_lengths)[::-1]
    probeImageName = os.path.join('/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/world/', qimgPath)
    ids_sorted_by_amount = ids[id_locs_sorted_by_amount_indexes]
    # probeim = cv2.imread(probeImageName)
    par_input = []
    # for i in id_locs_sorted_by_amount_indexes:
    #     imgID = ids[i]
    #     if imgID >= 0:
    #         dbImageName = IDtoImage[str(imgID)]
    #         locs = id_locs[i]
    for i in bar(id_locs_sorted_by_amount_indexes):
        imgID = ids[i]
        if imgID >= 0:
            dbImageName = IDtoImage[str(imgID)]
            dbImageName = os.path.join('/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/world/',dbImageName)
            # if imgID == 5284:#os.path.basename(dbImageName)  == 'c864857d798a2aabac634f4b0c91080c.jpg': #== '2fbde6ce8685469be654ace559df2483.jpg':#== '866d1f4f68e89f9c3060ba3c5eb329ee.jpg':
            #     print('stop')
            #if os.path.basename(dbImageName) == '334994c6b30cd5a71e5d5ec51723a832.jpg' or os.path.basename(dbImageName) == 'e0b102d0ca998f8f9ef35ff99af4c4df.jpg':
            # if os.path.basename(dbImageName) == '695c9885ebd2ee632b5daa56a3a18391.jpg':
            #     print('stop')
            # if os.path.basename(dbImageName) == 'f38133d72bf3be64889b6ea2d533604a.jpg':
            #     print('stop')
                #dbim = cv2.imread(dbImageName)
            dbim = 1#cv2.imread(dbImageName)
            locs = id_locs[i]
            # print('object points: ',len(locs))
            if len(locs) > 3:
                featureCoords = unravel_index(locs,I.shape)
                distances = D[featureCoords]
                meanDist = distances.mean()
                medianDist = np.median(distances)
                t = avgDist+1*distSTD
                if meanDist < t or medianDist < t:
                    imgSize =  imgIDtoShapeDict[imgID][::-1]
                    featureIDs_query = featureCoords[0]
                    featureIDs_query_unq,unique_featureIDs_query_inds = np.unique(featureIDs_query,return_index=True)
                    featureCoords_unq = (featureCoords[0][unique_featureIDs_query_inds],featureCoords[1][unique_featureIDs_query_inds])

                    featureIDs_database = featureLevel_I[featureCoords]
                    featureIDs_database_unq = featureLevel_I[featureCoords_unq]
                    featureIDs_database_unq, unique_featureIDs_database_inds = np.unique(featureIDs_database_unq,return_index=True)
                    featureCoords_unq = (featureCoords[0][unique_featureIDs_database_inds], featureCoords[1][unique_featureIDs_database_inds])
                    #featureIDs_query_unq = featureCoords[unique_featureIDs_query_inds[unique_featureIDs_database_inds]]
                    unique_indexes = unique_featureIDs_query_inds[unique_featureIDs_database_inds]
                    featureIDs_database_unq = featureLevel_I[featureCoords_unq]
                    #featureIDs_query = featureIDs_query[unique_featureIDs_database_inds]
                    featureScores = adaptiveScores[featureCoords].reshape((len(featureCoords[0]),1))
                    query_keypoint_metadata_local =  np.hstack((query_keypoint_metadata[featureIDs_query],featureIDs_query.reshape((len(featureIDs_query),1))))
                    featureLocations_inQuery = query_keypoint_metadata_local[:,0:2]
                    result_keypoint_metadata = np.hstack((database_keypoint_metadata[featureIDs_database,:-1],featureIDs_database.reshape((len(featureIDs_database),1))))
                    smin = featureScores.min()
                    if smin < 1:
                        featurescores_bias=(1-smin)+featureScores*featureScores
                    else:
                        featurescores_bias = featureScores
                    #centroid = (featureLocations_inQuery*featurescores_bias).mean(axis=0) # Defines the centroid of the object in question
                    # centroid = featureLocations_inQuery[(featureScores > featureScores.max() - featureScores.std() * 1).flatten()].mean(axis=0)

                    #t = np.zeros(featureLocations_inQuery.max(axis=0).astype(np.int)[::-1]+1)
                    #t[(featureLocations_inQuery[:,1].astype(np.int),featureLocations_inQuery[:,0].astype(np.int))] = (featureScores*featureScores).flatten()
                    #moments = cv2.moments(t)
                    #centroid = np.asarray([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']],dtype=np.float32)
                    featureScoresSquared = (featureScores*featureScores).flatten()
                    centroid = SCSM.centerOfMass(featureLocations_inQuery,featureScoresSquared)

                    #par_input.append((query_keypoint_metadata_local.astype(np.float32),result_keypoint_metadata.astype(np.float32),featureScores,centroid,imgSize,dbim))
                    score = SCSM.SCSM_kp(query_keypoint_metadata_local.astype(np.float32),result_keypoint_metadata.astype(np.float32),featureScores,centroid,imgSize,probeim,dbim,dbImageName)
                    scores_for_centroids.append(score)
                    ids_for_centroids.append(imgID)
        i+=1
#    if jobNum > 1:
#        startMultiThreadedProgress(len(par_input))
#        scores_for_centroids = Parallel(n_jobs=jobNum)(delayed(single_scsm_vote)(probeim,input) for input in par_input)
#        quitProg.value = False
#    else:
#        bar = progressbar.ProgressBar()
#        for input in bar(par_input):
#            scores_for_centroids.append(single_scsm_vote(probeim,input))
    scores_for_centroids = np.asarray(scores_for_centroids)
    sortedScoreIndexes = np.argsort(scores_for_centroids)[::-1]
    sortedIDs = np.aarallsarray(ids_for_centroids)[sortedScoreIndexes]
    sortedScores = scores_for_centroids[sortedScoreIndexes]
    return sortedIDs,sortedScores

def tallyVotes(D, featureLevel_I,imageLevel_I,numcores=1,useWeights = True):
    #I = featureIDtoImageID[featureLevel_I.flatten()]
    print(np.unique(imageLevel_I))
    print(D.flatten().shape,imageLevel_I.shape)
    I_unq,I_contig = np.unique(imageLevel_I,return_inverse=True)
    Dinv = 1 / (1 + D)
    weightedVotes = np.bincount(I_contig,weights=Dinv.flatten())
    maxVoteVal = weightedVotes.max()
    sortinds = weightedVotes.argsort()[::-1]
    sortedIDs = I_unq[sortinds]
    sortedVotes = weightedVotes[sortinds]
    return sortedIDs,sortedVotes,maxVoteVal

    Ishape = featureLevel_I.shape
    I = np.reshape(I,featureLevel_I.shape)
    ids, unq_inv, votes = np.unique(I,return_inverse=True, return_counts=True)
    I_contig = np.arange(len(ids))[unq_inv].astype(int)
    unq_inv_s = np.argsort(unq_inv)
    id_locs = np.split(unq_inv_s,np.cumsum(votes[:-1]))

    if useWeights:
        votes = np.bincount(I_contig,weights=Dinv.flatten())

    voteOrder = np.argsort(votes)[::-1]
    sortedIDs = ids[voteOrder]
    sortedVotes = votes[voteOrder]
    nIndex = np.where(sortedIDs == -1)[0]
    if len(nIndex) > 0:
        sortedIDs = np.delete(sortedIDs, nIndex[0])
        sortedVotes = np.delete(sortedVotes, nIndex[0])
    maxVoteVal = I.shape[0]* np.sum(1.0 /(np.arange(I.shape[1]) + 1))
    return sortedIDs, sortedVotes, maxVoteVal

def make_vres_vdev(gpu_resources,i0=0, i1=-1,ngpu=0,):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
    for i in range(int(i0), int(i1)):
        print("i0: " + str(i0) + "i1: " + str(i1))
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev

def featuresFitWithinRam(RAM,dims,float16,tempmem=0):
    ramBytes = float(RAM*1024*1024*1024)-tempmem
    dimBitSize = 32
    if float16:
        dimBitSize = 16
    featuresForRam = ramBytes/float(dims*(dimBitSize/8))
    return int(featuresForRam)

def wake_up_gpus(ngpu,tempmem):
    print("preparing resources for %d GPUs" % ngpu)
    gpu_resources = []
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)
    return gpu_resources

def loadIndexParameters(indexParametersResource):
  if indexParametersResource is not None:
      indexParameterData = indexParametersResource._data #Takes 150 Mb of memory
      bytearrays = fileutil.splitMultiFileByteArray(indexParameterData, 12, 5) #Takes another 150 Mb of memory
      tmppath = 'tmp'
      mv = memoryview(indexParameterData)
      all_tmp_paths = []
      count = 0
      params = []
      for bytearray,i in zip(bytearrays,range(len(bytearrays))):
          if i < 2:
              p = tmppath + str(count) + '.dat'
              with open(p, 'wb') as fp:
                  fp.write(mv[bytearray[0]:bytearray[1]])

              all_tmp_paths.append(p)
          else:
              sstring=bytes(mv[bytearray[0]:bytearray[1]]).decode('ascii')
              params.append(sstring)
          count += 1

      index = faiss.read_index(all_tmp_paths[0]) #WHY 12.5 GB?!!?!
      emptyIndexPath = all_tmp_paths[0]
      preproc = faiss.read_VectorTransform(all_tmp_paths[1])

      return(index,preproc,emptyIndexPath,params)
  else:
     indexParameterData=None
     return (None,None,None)
