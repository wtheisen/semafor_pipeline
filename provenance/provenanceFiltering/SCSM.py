print('starting SCSM import')

import sys
import os
import numpy as np
# import cv2 as cv
print('starting featureExtractor')
import featureExtractor
print(featureExtractor.__file__)
print('ended feature Extractor')
import matplotlib.pyplot as plt
import math
import scipy.stats as st
from scipy.sparse import lil_matrix
from scipy import signal
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
print('starting Dense Cluster Finder')
import DenseClusterFinder
import json
import time
randfiles = []
#plt = None
fig = None#plt.figure(figsize=(1,3))
pdf = st.norm.pdf
print('ended SCSM import')

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.max()
    return kernel

def pdfsum(X,nsig,nsize):
    return st.norm.pdf(X/nsize,scale=nsig)/st.norm.pdf(0,scale=nsig)
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy
def centerOfMass(points,weights = None):
    if weights is None:
        return points.sum(axis=0)/points.shape[0]
    return (points.T * weights).sum(axis=1) / weights.sum()
def apply_kernel_to_points(coords,scores,imgSize,kernel):
    map = np.zeros((imgSize[0]+kernel.shape[0],imgSize[1]+kernel.shape[1]))
    kernelysize=kernel.shape[0]
    kernelxsize=kernel.shape[1]
    halfnsizex=int(kernelxsize/2)
    halfnsizey=int(kernelysize/2)
    i=0
    for voteCoord in coords:
        voteScore = scores[i]
        y1=voteCoord[1]+halfnsizey
        y2 = y1+kernelysize
        x1=voteCoord[0]+halfnsizex
        x2=x1+kernelxsize
        m=map[y1:y2,x1:x2]
        map[y1:y2,x1:x2] += kernel[:m.shape[0],:m.shape[1]]*voteScore
        i+=1
    return map[halfnsizex:-halfnsizex,halfnsizey:-halfnsizey]



def SCSM_kp(query_meta,database_meta,matchScores,query_center,databaseShape,probeim,dbim,dbimname): #metadata is in matrix form, with rows as (x,y,scale,angle)
    showResults = False
    rotationAngles = (-query_meta[:,3]+database_meta[:,3])*math.pi/180
    scaleFactors = (database_meta[:,2]/query_meta[:,2]).reshape((query_meta.shape[0],1))
    #generate the negative (clockwise) rotation matrix to rotate our keypoints based on their orientation
    query_rotations1 = np.vstack((np.cos(rotationAngles),-np.sin(rotationAngles))).T
    query_rotations2 = np.fliplr(query_rotations1*np.asarray([1,-1]))
    #subtract the object center from the query keypoints, devide the remaining vector by the keypoint's scale
    adjusted_queryPoints = (query_center-query_meta[:,0:2])*scaleFactors
    #rotate the query keypoints clockwise
    initial_vectors = np.vstack((np.sum(adjusted_queryPoints*query_rotations1,axis=1),np.sum(adjusted_queryPoints*query_rotations2,axis=1))).T
    #Add the projected vectors calculated from the query back onto the matched database vectors
    voteCoords = database_meta[:,0:2]+initial_vectors # = adjusted_databasePoints
    adjusted_databasePoints = voteCoords

    vmShape = databaseShape[:2]
    # voteCoords += np.asarray([nsize / 2, nsize / 2], dtype=np.int)
    voteCoords = voteCoords.astype(np.int)
    return voteCoords
    goodvotes = np.bitwise_and(voteCoords[:, 0] >= 0, np.bitwise_and(voteCoords[:, 0] < vmShape[1],
                                                                     np.bitwise_and(voteCoords[:, 1] >= 0,
                                                                                    voteCoords[:, 1] <
                                                                                    vmShape[0])))
    # votemap_conv = np.zeros(databaseShape[:2], dtype=np.float32)
    # voteCoords = voteCoords[goodvotes]
    if len(voteCoords) > 0:

        vectorMagnitudes = np.linalg.norm(voteCoords, axis=1)
        vectorMagnitudes = vectorMagnitudes[np.logical_not(np.isnan(vectorMagnitudes))]
        meanVectorLength = vectorMagnitudes.mean()
        if math.isnan(meanVectorLength):
            meanVectorLength = 0
        nsize1 = min(meanVectorLength/(10*math.log10(meanVectorLength)),meanVectorLength/10)
        if meanVectorLength <= 100:
            nsize1 = meanVectorLength/10
        nsize = min(max(int(nsize1), 5),500)
        #nsize = math.log(meanVectorLength*meanVectorLength)

        useDBS = False
        # if useDBS:
        #     dbs = DBSCAN(nsize).fit(voteCoords)
        #     labels = dbs.labels_
        #     uniqueLabels = set(labels)
        # else:
        #     dbs = MeanShift().fit(voteCoords)
        #     labels = dbs.labels_
        #     uniqueLabels = set(labels)
        #n_clusters_ = len(uniqueLabels) - (1 if -1 in labels else 0)

        uniqueLabels,points,labels,indexesToUse = DenseClusterFinder.findBestDenseClusterUsingTable(voteCoords,nsize*3)
        if True:# n_clusters_ > 0:
            clusterImgScores = []
            # bestClusterFitness = float('inf') #lower the better
            for l in uniqueLabels:
                if l > -1:
                    #indexesToUse = labels == l
                    scoresInCluster = matchScores[indexesToUse]
                    coordsInCluster = voteCoords[indexesToUse]
                    scaleRatios = query_meta[:, 2] / database_meta[:, 2]

                    bbox = (coordsInCluster[:,0].min(),coordsInCluster[:,1].min(),coordsInCluster[:,0].max(),coordsInCluster[:,1].max())
                    bboxArea = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    imArea = databaseShape[0]*databaseShape[1]
                    if bboxArea < imArea:
                        clusterCenter = centerOfMass(coordsInCluster)
                        distsFromCenter = np.linalg.norm(coordsInCluster-clusterCenter,axis=1)
                        clusterFitness = distsFromCenter.mean()*math.log(len(scoresInCluster))
                        uniqueQueryIDs,uniqueQueryID_inds = np.unique(query_meta[indexesToUse,4],return_index=True)
                        uniqueDatabaseIDs,uniqueDatabaseID_inds = np.unique(database_meta[indexesToUse[uniqueQueryID_inds],4],return_index=True)

                        uniqueIndexesInCluster = indexesToUse[uniqueQueryID_inds[uniqueDatabaseID_inds]].astype(int)#np.intersect1d(indexesToUse,uniqueIndexes)
                        angleDifferences = abs(query_meta[uniqueIndexesInCluster, 3] - database_meta[uniqueIndexesInCluster, 3])
                        angleDifs_gt = [angleDifferences > 180]
                        angleDifferences[angleDifs_gt] = 360 - angleDifferences[angleDifs_gt]
                        angleDifferencesMean = angleDifferences.mean()
                        scaleRatios = query_meta[indexesToUse, 2] - database_meta[indexesToUse, 2]
                        angleDifSTD = angleDifferences.std()
                        scaleRatioSTD = scaleRatios.std()
                        bestClusterFitness = clusterFitness
                        voteScores = pdf(distsFromCenter/nsize,scale=3)/pdf(0,scale=3).flatten()
                        #chull = cv.convexHull(coordsInCluster)
                        #pointArea = cv.contourArea(chull)
                        # pointDensity = len(scoresInCluster)/(1+pointArea)#((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
                        #imgScore = scoresInCluster.sum()/len(scoresInCluster)*math.log10(len(scoresInCluster)) #exp1
                        #imgScore = scoresInCluster.sum() #exp2
                        #imgScore = scoresInCluster.sum()/(1+math.log(len(scoresInCluster))) #exp3
                        #imgScore = scoresInCluster.sum()*nsize/(1+.1*pointArea) #exp4
                        pointDensity_local = len(scoresInCluster)/(max(1,(bbox[2]-bbox[0])*(bbox[3]-bbox[1])))
                        pointDensity_global = (len(matchScores)-len(scoresInCluster))/max(1,databaseShape[0]*databaseShape[1]-(bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
                        angleCoherenceScore = 1/(1+angleDifSTD)
                        numberOfVotesScore = len(uniqueIndexesInCluster)/query_meta.shape[0]
                        #objectCoherenceScore = 1/(1+scaleRatioSTD/scaleRatios.mean())
                        #imgScore = (scoresInCluster.flatten()*voteScores).sum()*(pointDensity_local-pointDensity_global)#exp 5
                        imgScore = (scoresInCluster.flatten()*voteScores).sum()/math.log10(len(1+scoresInCluster))*1/(1+pointDensity_global/max(.00000000000001,pointDensity_local))*angleCoherenceScore*math.log10(len(scoresInCluster))*numberOfVotesScore #exp6
                        #imgScore = (scoresInCluster.flatten()*voteScores).sum()/len(scoresInCluster)*1/(1+angleDifSTD)*1/(1+scaleRatioSTD)
                        #if math.isnan(imgScore) or imgScore > .28:
                            #print(imgScore)
                        # imgScore = (scoresInCluster.flatten() * voteScores).sum()  # exp7 (did not exclude lower clusterFitness scores)
                        #imgScore = angleCoherenceScore*math.log(len(scoresInCluster))
                        clusterImgScores.append(imgScore)
            if len(clusterImgScores) > 0:
                bestScore = np.asarray(clusterImgScores).max()
            else:
                bestScore = 0
            if showResults:# and os.path.basename(dbimname) == '866d1f4f68e89f9c3060ba3c5eb329ee.jpg':#and os.path.basename(dbimname) == 'f4facd019c79d74d8cdc0c1ec263e82c.jpg':
                dbim = cv.imread(dbimname)
                kernel = gkern(nsize, 2)
                plt.figure(fig.number)
                cv.circle(probeim,tuple(query_center.astype(int)),30,(0,255,0),-1)
                fig.add_subplot(1,3,1)
                plt.imshow(probeim)
                fig.add_subplot(1, 3, 2)
                plt.imshow(dbim)
                fig.add_subplot(1, 3, 3)

                gvoteCoords = voteCoords[goodvotes]
                votemap_conv2 = apply_kernel_to_points(gvoteCoords, matchScores[goodvotes].flatten(), databaseShape[:2],
                                                       kernel)
                votemap_conv3 = np.zeros_like(votemap_conv2)
                votemap_conv3[(gvoteCoords[:,1],gvoteCoords[:,0])] = matchScores[goodvotes].flatten()
                votemap_conv4 = np.zeros_like(votemap_conv2)
                votemap_conv4[(gvoteCoords[:,1],gvoteCoords[:,0])] = 1
                votemap_conv5 = np.zeros((voteCoords[:,1].max(),voteCoords[:,0].max()))

                #img3 = cv.drawMatchesKnn(dbim, img1FeatureStuct[0], img2, img2FeatureStuct[0], goodMatches, None)
                plt.imshow(votemap_conv2)
                plt.show(block=False)
                plt.waitforbuttonpress()
                plt.clf()
                print('done')
            return bestScore
        else:
            return 0
        # votemap_conv[[voteCoords[:, 1], voteCoords[:, 0]]] = matchScores[goodvotes].flatten()

        # print('convolving...')
        # print('kernel size: ',kernel.shape)
        # votemap_conv = signal.fftconvolve(votemap_conv, kernel, mode='same')

        # votemap_conv2 = apply_kernel_to_points(voteCoords,matchScores[goodvotes].flatten(),votemap_conv.shape,kernel)


        originalVoteMax = votemap_conv.max()
        return votemap_conv2.max()
    return 0
    return adjusted_databasePoints
    ad_x = adjusted_databasePoints[:,0]
    ad_y = adjusted_databasePoints[:,1]
    minx = ad_x.min()
    maxx = ad_x.max()
    miny = ad_y.min()
    maxy = ad_y.max()
    voteMatwidth = maxx-minx
    voteMatheight = maxy-miny
    maxKernelSize = 1
    vectorMagnitudes = np.linalg.norm(adjusted_databasePoints, axis=1)
    meanVectorLength = vectorMagnitudes.mean()
    nsize = max(int(meanVectorLength / 12), 20)
    scale = maxKernelSize/nsize
    if voteMatheight*scale < 10 or voteMatwidth*scale < 10:
        scale = max(100/voteMatwidth,100/voteMatheight)
        maxKernelSize = scale*nsize
    nsize = maxKernelSize
    kernel = 1# gkern(nsize, 2)
    # kernel/=kernel.max()
    #bring the vote vectors' 0,0,origin to top left,with enough padding for possible gaussian kernel width
    adjusted_databasePoints = adjusted_databasePoints-np.asarray([minx,miny])
    halfnsize = int(nsize/2)
    # return 1
    voteMap = lil_matrix((int(nsize+1+voteMatheight*scale),int((nsize+1+voteMatwidth*scale))))
    i = 0
    voteCoordinates = (adjusted_databasePoints*scale).astype(np.int)+math.ceil(nsize/2)
    for voteCoord in (adjusted_databasePoints*scale).astype(np.int)+math.ceil(nsize/2):
        print(i)
        voteScore = matchScores[i][0]
        y1=voteCoord[1]-halfnsize
        y2 = y1+nsize
        x1=voteCoord[0]-halfnsize
        x2=x1+nsize
        voteMap[y1:y2,x1:x2] += kernel/(1+voteScore)
        i+=1
    voteMap_Nonsparse = voteMap.tocoo()
    maxVoteScore = voteMap_Nonsparse.max()
    finalVote = maxVoteScore/adjusted_databasePoints.shape[0]*math.log10(adjusted_databasePoints.shape[0])
    return finalVote


def SCSM(img1,img1Rect,img2,autoCenter=True):
    if not autoCenter:
        roi = img1[img1Rect[1]:img1Rect[1]+img1Rect[3],img1Rect[0]:img1Rect[0]+img1Rect[2]]
    else:
        roi = img1
    bestMapScore = 0
    bestMap = 0
    bestPoints = None
    centerxBest = None
    centeryBest = None
    wasflipped = False
    for flip in [False,True]:
        if flip:
            img2 = np.fliplr(img2)

        #roi = cv.flip(roi,1)
        # Extract Features
        #print('extracting features...')
        t0 = time.time()
        img1FeatureStuct = featureExtractor.local_feature_detection_and_description("", 'SURF', 'SURF',
                                                                                  5000, roi, mask=None,
                                                                                  dense_descriptor=False,
                                                                                  default_params=True)
        img2FeatureStuct = featureExtractor.local_feature_detection_and_description("", 'SURF', 'SURF',
                                                                                    5000, img2, mask=None,
                                                                                    dense_descriptor=False,
                                                                                    default_params=True)
        t1 = time.time()
        #print('generating vectors...')
        img1Vectors = []
        f1 = img1FeatureStuct[1]
        f2  = img2FeatureStuct[1]
        # Get the roi center coordinates

        # subtract keypoint vectors from roi center so we can map matched keypoints from the database image to an "object center", divide out feature magnitude, subtract out the angle


        bf = cv.BFMatcher()
        allMatches = bf.knnMatch(f1,f2,k=2)
        goodMatches = []
        goodMatchDists = []
        xpoints = []
        ypoints = []
        i=0
        matchToQ = {}
        useOneToOne=False
        for m, n in allMatches:
            if m.distance < .75 * n.distance :#or True:
                #goodMatches.append([m])
                #goodMatchDists.append(m.distance)
                qkp = img1FeatureStuct[0][m.queryIdx]
                dkp = img2FeatureStuct[0][m.trainIdx]
                if dkp in matchToQ:
                    if matchToQ[dkp].distance > m.distance:
                        matchToQ[dkp] = m
                else:
                    matchToQ[dkp] = m
                if not useOneToOne:
                    goodMatchDists.append(m.distance)
                    goodMatches.append([m])
                    xpoints.append(qkp.pt[0])
                    ypoints.append(qkp.pt[1])
            i+=1
        if useOneToOne:
            for dkp in matchToQ:
                m = matchToQ[dkp]
                goodMatches.append([m])
                goodMatchDists.append(m.distance)
                qkp = img1FeatureStuct[0][m.queryIdx]
                xpoints.append(qkp.pt[0])
                ypoints.append(qkp.pt[1])

        #goodMatches = allMatches
        goodMatchDists = np.asarray(goodMatchDists)
        xpoints = np.array(xpoints)
        ypoints = np.array(ypoints)
        maxDist = goodMatchDists.max()
        minDist = goodMatchDists.min()
        normDists = (goodMatchDists-minDist)/(maxDist-minDist)
        matchim2 = np.array([])
        matchim2 = cv.drawMatches(img1, img1FeatureStuct[0], img2, img2FeatureStuct[0],
                                   [match[0] for match in goodMatches], matchim2)
        if autoCenter or not flip:
            goodMatchDistsInv = 1 / (1 + normDists)
            centerX = (xpoints*goodMatchDistsInv).sum()/goodMatchDistsInv.sum()
            centerY = (ypoints*goodMatchDistsInv).sum()/goodMatchDistsInv.sum()
            stdX = xpoints.std()
            stdY = ypoints.std()
            z=2
            roi1 = [centerX-(stdX*z)/2,centerY-stdY*z/2,stdX*z,stdY*z]
            g=np.bitwise_and(np.bitwise_and(xpoints >= roi1[0],xpoints < roi1[0]+roi1[2]),np.bitwise_and(ypoints >= roi1[1],ypoints < roi1[1]+roi1[2]))
            xpoints2 = xpoints[g]
            ypoints2 = ypoints[g]
            centerX2 = (xpoints2 * goodMatchDistsInv[g]).sum() / goodMatchDistsInv[g].sum()
            centerY2 = (ypoints2 * goodMatchDistsInv[g]).sum() / goodMatchDistsInv[g].sum()
            roiCenter = (centerX2,centerY2)
        else:
            roiCenter = (191,710)#(img1Rect[2] / 2, img1Rect[3] / 2)
            centerX2 = 191
            centerY2 = 710
        for kp in img1FeatureStuct[0]:
            pt = kp.pt
            fangle = kp.angle*math.pi/180
            cvector=((roiCenter[0]-pt[0])/kp.size,(roiCenter[1]-pt[1])/kp.size)
            mag,ang = cart2pol(cvector[0],cvector[1])
            cvector = pol2cart(mag,ang-fangle)
            img1Vectors.append(cvector)

        matchScores = 1-normDists
        voteMap = np.zeros((img2.shape[0],img2.shape[1]),dtype=np.float32)
        nsize = int(math.pow(max(img2.shape[0],img2.shape[1])/6,.9))
        i = 0
        kernel = gkern(nsize,2)
        queryMeta = []
        databaseMeta = []
        scores = []
        count = 0
        from sklearn.cluster import DBSCAN
        for m in goodMatches:
            qkp = img1FeatureStuct[0][m[0].queryIdx]
            dkp = img2FeatureStuct[0][m[0].trainIdx]
            queryMeta.append([qkp.pt[0],qkp.pt[1],qkp.size,qkp.angle,m[0].queryIdx])
            databaseMeta.append([dkp.pt[0],dkp.pt[1],dkp.size,dkp.angle,m[0].trainIdx])
            scores.append(1/(1+m[0].distance))
            count += 1

        queryMeta = np.asarray(queryMeta)
        databaseMeta = np.asarray(databaseMeta)
        scores = np.asarray(scores)
        #scores = (scores-scores.min())/(scores.max()-scores.min())
        voteCoords = (SCSM_kp(queryMeta,databaseMeta,scores,roiCenter,img2.shape,None,None,"")-nsize/2).astype(np.int)
        voteMap_round = voteMap.copy()
        for voteCoord in voteCoords:
            if voteCoord[0] >= 0 and voteCoord[0]+nsize < img2.shape[1] and voteCoord[1] >= 0 and voteCoord[1]+nsize < img2.shape[0]:
                voteMap_round[voteCoord[1]:voteCoord[1]+nsize,voteCoord[0]:voteCoord[0]+nsize] += kernel*scores[i]
            i+=1
        score = voteMap_round.max()
        voteCenter = np.unravel_index(voteMap_round.flatten().argmax(),voteMap.shape)[::-1]
        i=0
        pdffunc = st.norm.pdf
        for voteCoord in voteCoords:
            windowsize = math.pow(max(img2.shape[0],img2.shape[1])/6,.9)
            radius = np.linalg.norm(voteCoord-voteCenter)
            stepsize = 1
            xrng = np.arange(stepsize, radius, stepsize)
            pf = pdffunc(xrng / windowsize, scale=1.5) / pdffunc(0, scale=1.5)
            pf = (pf - pf.min()) / (1 - pf.min())
            linkern = np.concatenate((pf[::-1], [1], pf)).reshape((1, -1))
            kern = linkern.T * linkern
            startx = int(voteCoord[0] - kern.shape[1] / 2)
            starty = int(voteCoord[1] - kern.shape[0] / 2)
            endx = startx + kern.shape[1]
            endy = starty + kern.shape[0]
            neededShape = voteMap[starty:endy, startx:endx].shape
            voteMap[starty:endy, startx:endx] += (kern * scores[i])[:neededShape[0], :neededShape[1]]
            i+=1
        if score > bestMapScore:
            bestMapScore = score
            bestMap = voteMap
            centerxBest = centerX2
            centeryBest = centerY2
            bestPoints = voteCoords
            wasflipped = flip

    if wasflipped:
        bestMap = np.fliplr(bestMap) #Only flip the vote map, the centers are still correct
        #centerxBest = img1.shape[1]-centerxBest
        #centeryBest = img1.shape[0]-centeryBest
    return bestMap,(centerxBest,centeryBest),score,t1-t0

    voteMap_KP = voteMap.copy()

    voteMap = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.float32)

    i=0
    for m in goodMatches:
        im1KpIndex = m[0].queryIdx
        im2KpIndex = m[0].trainIdx
        kp = img2FeatureStuct[0][im2KpIndex]
        kpLoc = kp.pt
        kpAng = kp.angle*math.pi/180
        cvector = img1Vectors[im1KpIndex]
        mag,ang = cart2pol(cvector[0],cvector[1])
        cvector = pol2cart(mag,ang+kpAng)
        voteCoord = (int((kpLoc[0]+cvector[0]*kp.size)-nsize/2),int((kpLoc[1]+cvector[1]*kp.size)-nsize/2))
        if voteCoord[0] >= 0 and voteCoord[0]+nsize < img2.shape[1] and voteCoord[1] >= 0 and voteCoord[1]+nsize < img2.shape[0]:

            voteMap[voteCoord[1]:voteCoord[1]+nsize,voteCoord[0]:voteCoord[0]+nsize] += (kernel*scores[i])#1/(1+normDists[i])
        # if i > 0:
        #     plt.close()
        #     t = img2.copy()
        #
        #     plt.imshow(voteMap)
        #     plt.savefig('./scsm_gif/threshold/'+str(i).zfill(3)+'.png')
        # print(i)
        i+=1
    voteMax = voteMap.max()
    voteMap_norm = voteMap/voteMax
    # img3 = cv.drawMatchesKnn(roi,img1FeatureStuct[0],img2,img2FeatureStuct[0],goodMatches,None)
    voteMin = voteMap.min()
    votemap_conv = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.float32)
    voteCoords += np.asarray([nsize/2,nsize/2],dtype=np.int)
    goodvotes = np.bitwise_and(voteCoords[:,0] >= 0,np.bitwise_and(voteCoords [:,0] < votemap_conv.shape[1],np.bitwise_and(voteCoords[:,1] >= 0, voteCoords[:,1] < votemap_conv.shape[0])))
    voteCoords = voteCoords[goodvotes]
    votemap_conv[[voteCoords[:, 1], voteCoords[:, 0]]] = scores[goodvotes]
    print('convolving...')
    votemap_conv = signal.fftconvolve(votemap_conv,kernel,mode='same')
    # steps = 100
    # step = (voteMax-voteMin)*1.0/steps
    # for i in range(0,steps):
    #     t = voteMap.copy()
    #     t[voteMap < voteMin+step*i]=0
    #     plt.imshow(t)
    #     plt.savefig('./scsm_gif/threshold/'+str(i+500).zfill(3)+'.png')
    #     plt.close()
    #     print(i)
    # plt.figure()
    # plt.imshow(voteMap)
    # plt.figure()
    # plt.imshow(voteMap_KP)
    # plt.figure()
    # plt.imshow(votemap_conv)
    # plt.show()
    return voteMap
    print('done')
# im1path = '/home/jbrogan4/Downloads/objDet/objDet/4109262011d27322377a02103d2a7259.jpg'
# im2path = '/home/jbrogan4/Downloads/objDet/objDet/test5.jpg'
# im1 = cv.imread(im1path)
# im2 = cv.imread(im2path)
# SCSM(im1,(2833,1183,704,1028),im2)

# im1path = '/home/jbrogan4/Documents/Projects/Medifor/tensorflow/models/research/delf/delf/python/examples/sqirrel.jpg'
# im2path = '/home/jbrogan4/Documents/Projects/Medifor/tensorflow/models/research/delf/delf/python/examples/snowwhite.jpg'
# im1 = cv.imread(im1path)
# im2 = cv.imread(im2path)
# SCSM(im1,(0,0,im2.shape[0],im2.shape[1]),im2)


def runSCSMForFileList(probeFile,fileList,outfolder,randomFiles=[]):
    try:
        #probeFile = '/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/g121_dcmfmii.jpg'
        probeImg = cv.imread(probeFile)
        print('reading ',probeFile)

        rect = (0,0,probeImg.shape[0],probeImg.shape[1])
        folderPath = os.path.dirname(probeFile)
        allImageFiles = os.listdir(folderPath)
        centerColors = [(113,213,121),(247,0,255),(0,0,0),(76,190,251),(246,47,0),(17,12,162),(113,213,121),(255,255,255),(247,50,255)]
        allCenters = []
        imgNames= []
        #fileList.append('/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/g121_tcuPvhq.png')
        #fileList.append('/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/g121_bbeCOhh.png')
        #fileList.append('/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/g121_337182.png')
        #fileList.append('/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/g121_zu8b21.jpg')
        #fileList.append('/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/g121_root.jpg')
        for r in randomFiles:
            fileList.append(r)
        scoresForImages = []
        allimages = []
        times= []
        for imgFile in fileList:
            #imgFile = os.path.join(folderPath,imgFile)
            if imgFile.endswith('.png') or imgFile.endswith('.jpg'):
                rImg = None
                try:
                    rImg = cv.imread(imgFile)
                except:
                    continue
                if rImg is None or rImg.shape[0] <= 0 or rImg.shape[1] <= 0:
                    print('image is none')
                else:
                    print(os.path.basename(imgFile))
                    if 'f52220'  in os.path.basename(imgFile):
                        print('pause')
                        continue
                    else:
                        pass
                    ac = True
                    if 'g121_bbeCOhh.png' in imgFile:
                        ac=False
                    votemap,center,score,time = SCSM(probeImg,rect,rImg,autoCenter=ac)
                    times.append(time)
                    scoresForImages.append(score)
                    allCenters.append(center)
                    vmax = votemap.max()
                    vmin = votemap.min()
                    votemap_norm = np.power(((votemap-vmin)/(vmax-vmin)),1.5)
                    #votemap_norm = ((votemap - vmin) / (vmax - vmin))
                    if 'root' in imgFile:
                        print('root')
                    outfile = os.path.join(outfolder,os.path.basename(imgFile))
                    allimages.append(imgFile)
                    cmap = plt.cm.jet
                    norm = plt.Normalize(vmin=votemap_norm.min(), vmax=votemap_norm.max())
                    cmapimg = (cmap(np.square(norm(votemap_norm)))[:,:,:-1]*255).astype(np.uint8)
                    overlay = cmapimg.copy()
                    output= rImg.copy()
                    cv.addWeighted(overlay, .5, rImg, .5, 0, output)
                    imgNames.append(os.path.basename(imgFile))
                    #plt.imsave(outfile+'.jpg', image)
                    outpathfile = os.path.basename(outfile+'.jpg').split('_')[0]
                    infolder = os.path.dirname(probeFile).split('/')[-1]
                    outfile2 = os.path.join(outfolder,infolder)
                    print(infolder,outfile2)
                    try:
                        os.makedirs(outfile2)
                    except:
                        pass
                    outfile = os.path.join(outfile2,os.path.basename(outfile))
                    outprobefile = os.path.join(outfile2,'query_'+os.path.basename(outfile))
                    print(outfile)
                    print(outprobefile)
                    outprobe = probeImg.copy()

                    outprobe = cv.circle(outprobe,(int(center[0]),int(center[1])),int(max(outprobe.shape[0],outprobe.shape[1])/35),(255,0,0),-1)
                    cv.imwrite(outfile+'.jpg',output)
                    cv.imwrite(outprobefile+'.jpg',outprobe)

    except:
        pass

    scoresForImages = np.array(scoresForImages)
    sortedinds = scoresForImages.argsort()[::-1]
    sortedscores = scoresForImages[sortedinds]
    filenames = np.array(allimages)[sortedinds]
    idwants=[]
    #imagesIwant = np.array(
    #    ['g121_337182.png', 'g121_bbeCOhh.png', 'g121_tcuPvhq.png', 'g121_dcmg8g1.png', 'g121_dcmq7bo.jpg',
    #     'g121_dcmjl7o.png','g121_root.jpg'])
    #for im in imagesIwant:
    #    p = os.path.join('/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/', im)
    #    id = np.where(filenames == p)
    #    idwants.append(id[0][0])
    print('done')

def runSCSMForFolder(folder,outfolder):
    if os.path.isdir(folder):
        files = os.listdir(folder)
        probeFile = None
        fileList = []
        for f in files:
            p = os.path.join(folder,f)
            if 'probe' in f:
                probeFile = p
            fileList.append(p)
        try:
            os.makedirs(outfolder)
        except:
            pass

        if probeFile is None:
            probeFile = fileList[0]
        runSCSMForFileList(probeFile, fileList, outfolder)

def runSCSMForJSON(jsonFile,outfolder):
    #folderPath = '/media/jbrogan4/scratch2/google-landmarks-dataset/train4/'
    folderPath = '/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/jsons'
    with open(jsonFile,'r') as fp:
        j = json.load(fp)
    if '26ac63' not in j['graph']['name']:
        print('skipping')
        return
    nodes = j['nodes']
    fileList = []
    probeFile = None

    for n in nodes:
        filename = os.path.basename(n['file'])
        dname = os.path.basename(os.path.dirname(n['file']))
        p = os.path.join((os.path.dirname(n['file'])[:-len(dname)]), 'Data', dname, filename)
        if os.path.isfile(p):
            #fileList.append(os.path.join(folderPath,n['file']))
            fileList.append(p)
            randfiles.append(p)
            #if filename.startswith(j['graph']['name']):
                #probeFile = p
    try:
        os.makedirs(outfolder)
    except:
        pass

    if probeFile is None:
        probeFile = fileList[0]
    randtosend = []
    if len(randfiles) > 3:
        randtosend = np.random.choice(len(randfiles),3)

    runSCSMForFileList(probeFile,fileList[1:],outfolder,randomFiles=randtosend)

if __name__ == "__main__":
    jsonFolder = sys.argv[1]
    outFolder = sys.argv[2]

    allJsons = os.listdir(jsonFolder)

    for j in allJsons:
        jsonFile = os.path.join(jsonFolder,j)
        #runSCSMForJSON(jsonFile,outFolder)
        runSCSMForFolder(jsonFile,outFolder)
