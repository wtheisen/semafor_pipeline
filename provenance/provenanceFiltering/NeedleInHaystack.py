import json
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import cv2
import progressbar
import scipy.stats as st
import collections
from joblib import Parallel,delayed

def runTally(I, Dinv, image_level_IDs):
    uniqueImageIDs, uniqueImageIDs_inds, uniqueImageIDs_inv, uniqueImageIDs_counts = np.unique(
        image_level_IDs.flatten(), return_index=True, return_inverse=True, return_counts=True)
    # imageIDtoContiguous = np.arange(len(uniqueImageIDs))[uniqueImageIDs_inv] #Map all of the image IDs to a contiguous set, easier for bin counting
    # flatIDs = image_level_IDs.flatten()
    # flatIDs_contig = imageIDtoContiguous
    # weightedVotes_for_contigIDs = np.bincount(flatIDs_contig,weights=Dinv.flatten())
    return uniqueImageIDs_counts, uniqueImageIDs


def performVQ(voteCoordstmp, qsize_matrix, shift, image_level_IDs, IDs_contig, featIDs, removeMultiMatch=False,
              removeMultiDatabase=False):
    voteCoords = voteCoordstmp.copy()
    # voteCoord_arr = voteCoords.reshape((-1,2))
    if shift == 'right':
        voteCoords = voteCoords.copy()
        voteCoords[:, :, 0] = (voteCoords[:, :, 0] - qsize_matrix[:, :, 0] / 2)
    if shift == 'down':
        voteCoords = voteCoords.copy()
        voteCoords[:, :, 1] = (voteCoords[:, :, 1] - qsize_matrix[:, :, 0] / 2)
    if shift == 'rightdown':
        voteCoords = voteCoords.copy()
        voteCoords[:, :, 0] = (voteCoords[:, :, 0] - qsize_matrix[:, :, 0] / 2)
        voteCoords[:, :, 1] = (voteCoords[:, :, 1] - qsize_matrix[:, :, 0] / 2)
    vq = (voteCoords / qsize_matrix).astype(np.int)
    vq[:, :, 0][voteCoords[:, :,
                0] < 0] -= 1  # To prevent all positive and negative points near zero bin from quantizing to the same
    vq[:, :, 1][voteCoords[:, :, 1] < 0] -= 1
    vq_arr = np.dstack((vq, IDs_contig)).reshape((-1, 3))
    minBins = vq_arr.min(axis=0)
    vq_arr = (vq_arr - minBins)
    maxBins = vq_arr.max(axis=0)
    vq_bin_ids = np.ravel_multi_index(vq_arr.T, (maxBins[0] + 1, maxBins[1] + 1, IDs_contig.max() + 1))
    IDs_contig_flat = IDs_contig.flatten()
    IDs_flat = image_level_IDs.flatten()
    maxbin = vq_bin_ids.max()
    otm_ids_unique_inds = np.arange(0, len(image_level_IDs))
    # original_image_level_IDs = image_level_IDs.copy()
    if removeMultiMatch:
        # remove items from bins that are all matching to the same query feature
        queryIDsMat = np.arange(0, qsize_matrix.shape[0]).reshape((-1, 1)).repeat(qsize_matrix.shape[1], axis=1)
        # 1-to-many ID
        queryIDs_and_binIDs = np.vstack((queryIDsMat.flatten(), vq_bin_ids)).astype(
            int)  # if features in a cluster match to the same query feature, that is 1-to-many matching, and we want to filter them out
        otm_ids = np.ravel_multi_index(queryIDs_and_binIDs, (qsize_matrix.shape[0], int(maxbin) + 1))
        # print(otm_ids[image_level_IDs==idToTest])
        # we know that the first occurence of a bin will have the highest match value, and that is the one we want to keep
        otm_ids_unique, otm_ids_unique_inds = np.unique(otm_ids, return_index=True)
        # otm_ids_unique_inds is the indexes of all of the feature matches we want to keep!
        # print('removed ', len(otm_ids)-len(otm_ids_unique), 'one-to-many matches')
        # print(voteCoords.shape,len(otm_ids))
        vq_bin_ids = vq_bin_ids[otm_ids_unique_inds]
        # voteCoords = voteCoords[otm_ids_unique_inds]
        vq_arr = vq_arr[otm_ids_unique_inds]
        IDs_contig_flat = IDs_contig_flat[otm_ids_unique_inds]
        IDs_flat = IDs_flat[otm_ids_unique_inds]
        featIDs = featIDs[otm_ids_unique_inds]
        # matrixInds = matrixInds[otm_ids_unique_inds]

    if removeMultiDatabase:
        # remove items from bins that are all matching to the same query feature
        # 1-to-many ID
        uniqueFeatIDs, uniqueFeatIDs_inv = np.unique(featIDs, return_inverse=True)
        featIDs_contig = np.arange(0, len(uniqueFeatIDs))[uniqueFeatIDs_inv]
        queryIDs_and_binIDs = np.vstack((featIDs_contig, vq_bin_ids)).astype(
            int)  # if features in a cluster match to the same query feature, that is 1-to-many matching, and we want to filter them out
        otm_ids = np.ravel_multi_index(queryIDs_and_binIDs, (featIDs_contig.max() + 1, int(maxbin) + 1))
        # we know that the first occurence of a bin will have the highest match value, and that is the one we want to keep
        otm_ids_unique, otm_ids_unique_inds2 = np.unique(otm_ids, return_index=True)
        # otm_ids_unique_inds is the indexes of all of the feature matches we want to keep!
        # print('removed ', len(otm_ids)-len(otm_ids_unique), 'one-to-many matches')
        # print(voteCoords.shape,len(otm_ids))
        vq_bin_ids = vq_bin_ids[otm_ids_unique_inds2]
        # voteCoords = voteCoords[otm_ids_unique_inds]
        vq_arr = vq_arr[otm_ids_unique_inds2]
        IDs_contig_flat = IDs_contig_flat[otm_ids_unique_inds2]
        IDs_flat = IDs_flat[otm_ids_unique_inds2]
        otm_ids_unique_inds = otm_ids_unique_inds[otm_ids_unique_inds2]
        # matrixInds = matrixInds[otm_ids_unique_inds]

    uniqueBins, bin_inds, bin_inv, bin_counts = np.unique(vq_bin_ids, return_index=True, return_counts=True,
                                                          return_inverse=True)

    counts_thres_inds = bin_counts > 0
    inv_thres_inds = bin_counts[bin_inv] > 0

    # remove items from bins that are all matching to the same query feature

    ids_for_bins = IDs_contig_flat.flatten()[bin_inds][counts_thres_inds]
    bin_counts_thresh = bin_counts[counts_thres_inds]
    uniqueBins_thresh = uniqueBins[counts_thres_inds]
    count_max = bin_counts_thresh.max()
    counts_and_ids = np.vstack((bin_counts_thresh, ids_for_bins))
    sorted_lex = np.lexsort(counts_and_ids)[::-1]
    counts_and_ids = counts_and_ids.T[sorted_lex]
    uniqueImageswithClusters, image_start_inds = np.unique(counts_and_ids[:, 1], return_index=True)
    clusterSortInds = image_start_inds[::-1]
    vq_ids_inv_sortedinds = np.argsort(bin_inv)
    imageids_forsortedinds = IDs_flat[vq_ids_inv_sortedinds]
    vqinv_and_imids = np.vstack((otm_ids_unique_inds[vq_ids_inv_sortedinds], imageids_forsortedinds)).T
    max_clusters_for_images = counts_and_ids[clusterSortInds]

    vq_ids_inv_sortedinds = vq_bin_ids[vq_ids_inv_sortedinds]

    idSplitList_raw = np.cumsum(bin_counts[:-1])
    points_of_clusters_raw = np.asarray(np.split(vqinv_and_imids,
                                                 idSplitList_raw))  # Produces a jagged array of indexes, in the order of uniqueBins_thresh
    reorder_inds = np.arange(len(bin_counts))[counts_thres_inds][sorted_lex][clusterSortInds]
    points_of_clusters = points_of_clusters_raw[reorder_inds]
    # size_of_clusters = np.array([p.shape[0] for p in points_of_clusters])
    # test = np.concatenate(points_of_clusters)
    # print('clusterArrSizes: ',vqinv_and_imids.shape,points_of_clusters_raw[10].shape,test.shape)
    # print('number of clusters: ',len(points_of_clusters))
    # print('number of vq inds: ', len(np.unique(IDs_contig_flat[test])))
    # print('poc: ',original_image_level_IDs.flatten()[test[:,0]].flatten()[-35:])
    # print('poc: ',test[:,1].flatten()[-35:])
    # print('original: ',original_image_level_IDs.flatten()[otm_ids_unique_inds[test[:,0]].flatten()][-35:])
    # print('sizes:',bin_counts[reorder_inds][-35:])
    # print(bin_counts[reorder_inds].max())
    return (points_of_clusters, bin_counts[reorder_inds])


def runDensityClustering(voteCoords, qsize_matrix, image_level_IDs, IDs_contig, featIDs, removeMultiMatch=False,
                         removeMultiDatabase=False):
    points_of_clusters = None
    size_of_clusters = None
    imids_for_clusters = None
    points_of_clusters = None
    size_of_clusters = None
    imids_for_clusters = None
    shiftList = ['None', 'right', 'down', 'rightdown']
    allPointsShiftedClusters = Parallel(n_jobs=len(shiftList))(delayed(performVQ)(voteCoords, qsize_matrix, offset, image_level_IDs, IDs_contig, featIDs,removeMultiMatch=removeMultiMatch, removeMultiDatabase=removeMultiDatabase) for offset in shiftList)
    for points_of_clusterst, size_of_clusterst in allPointsShiftedClusters:
        if points_of_clusters is None:
            points_of_clusters = points_of_clusterst.copy()
            size_of_clusters = size_of_clusterst.copy()
        else:
            betterSizes = (size_of_clusterst - size_of_clusters) > 0
            size_of_clusters[betterSizes] = size_of_clusterst[betterSizes]
            points_of_clusters[betterSizes] = points_of_clusterst[betterSizes]

    # for offset in ['None', 'right', 'down', 'rightdown']:
    #     points_of_clusterst, size_of_clusterst = performVQ(voteCoords, qsize_matrix, offset, image_level_IDs,
    #                                                        IDs_contig, featIDs, removeMultiMatch=removeMultiMatch,
    #                                                        removeMultiDatabase=removeMultiDatabase)
    #     if points_of_clusters is None:
    #         points_of_clusters = points_of_clusterst.copy()
    #         size_of_clusters = size_of_clusterst.copy()
    #     else:
    #         betterSizes = (size_of_clusterst - size_of_clusters) > 0
    #         size_of_clusters[betterSizes] = size_of_clusterst[betterSizes]
    #         points_of_clusters[betterSizes] = points_of_clusterst[betterSizes]
    return points_of_clusters, size_of_clusters


# points_of_clusters,size_of_clusters = runDensityClustering(voteCoords_final,qsize_matrix,IDs_contig)
def getQuantizationSizes(imageSizes, divis=6, power=.9):
    qsize = np.power(np.amax(imageSizes, axis=1) / divis, power)
    qsize = np.minimum(np.maximum(qsize.astype(np.int), 5), 250)
    # qsize_matrix = qsize.reshape((image_level_IDs.shape[0], -1, 1))
    return qsize  # ,qsize_matrix


def Dscore(D):
    D2 = 1 / (1 + np.sqrt(D))

    # D2 = D.copy()
    D2[np.isinf(D)] = np.nan
    D2[D2 > 100] = np.nan
    return D2
    Dstd = np.nanstd(D2)
    invD = -D + (np.median(D) + Dstd * 3)
    invD[invD < 0] = .00000000001
    del D2
    return invD


def projectVotes(I, invD, image_level_IDs, allMetaDatabase, allMetaQuery, imageSizes):
    rotationAngles = (-allMetaQuery[:, :, 3] + allMetaDatabase[:, :, 3]) * math.pi / 180
    scaleFactors = ((allMetaDatabase[:, :, 2] / allMetaQuery[:, :, 2])).reshape((allMetaQuery.shape[0], -1, 1))

    # invD = Dscore(D)

    # Calculate centroids for each image ID
    uniqueImageIDs, uniqueImageIDs_inds, uniqueImageIDs_inv, uniqueImageIDs_counts = np.unique(
        image_level_IDs.flatten(), return_index=True, return_inverse=True, return_counts=True)
    #     uniqueImageIDs_reordered,uniqueImageIDs_reordered_inds,uniqueImageIDs_reordered_inv = np.unique(uniqueImageIDs,return_index=True,return_inverse=True)
    imageIDtoContiguous = np.arange(len(uniqueImageIDs))[
        uniqueImageIDs_inv]  # Map all of the image IDs to a contiguous set, easier for bin counting
    flatIDs = image_level_IDs.flatten()
    flatIDs_contig = imageIDtoContiguous
    IDs_contig = flatIDs_contig.reshape(I.shape)
    goodInds = flatIDs >= 0
    # get back to original image IDs by uniqueImageIDs[contigID]

    flatIDs_good = flatIDs[goodInds]
    uniqueImageIDs_inv_good = uniqueImageIDs_inv[goodInds]
    image_centroid_weights = np.bincount(flatIDs_contig, weights=invD.flatten())
    centroid_weight_matrix = image_centroid_weights[uniqueImageIDs_inv].reshape(image_level_IDs.shape)
    # find centroids for each image ID, put in array ordered by uniqueImageIDs
    all_centroids_x = np.bincount(flatIDs_contig,
                                  weights=(allMetaQuery[:, :, 0] * invD).flatten()) / image_centroid_weights
    all_centroids_y = np.bincount(flatIDs_contig,
                                  weights=(allMetaQuery[:, :, 1] * invD).flatten()) / image_centroid_weights
    all_centroids = np.vstack((all_centroids_x, all_centroids_y)).T  # Should by a (#of unique images * 2) sized matrix
    centroid_application_matrix = all_centroids[uniqueImageIDs_inv].reshape((image_level_IDs.shape[0],
                                                                             image_level_IDs.shape[1],
                                                                             2))  # Should be a tensor of original match matrix size, with x and y channel

    # Calculate vote coordinates in hough space based on metadata from matched keypoints
    query_rotations1 = np.concatenate((np.cos(rotationAngles).reshape((rotationAngles.shape[0], -1, 1)),
                                       -np.sin(rotationAngles).reshape((rotationAngles.shape[0], -1, 1))), axis=2)
    query_rotations2 = np.flip(query_rotations1 * np.asarray([[[1, -1]]]), axis=2)
    adjusted_queryPoints = (centroid_application_matrix - allMetaQuery[:, :, 0:2]) * scaleFactors  # has x and y channel
    initial_vectors = np.concatenate((
                                     (adjusted_queryPoints * query_rotations1).sum(axis=2).reshape((I.shape[0], -1, 1)),
                                     (adjusted_queryPoints * query_rotations2).sum(axis=2).reshape(I.shape[0], -1, 1)),
                                     axis=2)
    voteCoords_final = allMetaDatabase[:, :, 0:2] + initial_vectors

    # vectorMagnitudes = np.linalg.norm(voteCoords_final,axis=2)
    # meanVectorLengths = np.bincount(flatIDs_contig,weights=vectorMagnitudes.flatten())/uniqueImageIDs_counts
    qsize = getQuantizationSizes(imageSizes)
    qsize_matrix = qsize.reshape((image_level_IDs.shape[0], -1, 1))

    # points_of_clusters,size_of_clusters = runDensityClustering(voteCoords_final,qsize_matrix,IDs_contig)
    # all_vq_bin_ids = np.concatenate(points_of_clusters)
    # NHScore_Vectorized(voteCoords_final,all_vq_bin_ids,Dinv,allMetaDatabase,allMetaQuery)
    return voteCoords_final, IDs_contig, centroid_application_matrix


def NHScore_Vectorized(voteCoords, points_of_clusters, Dinv, allMetaDatabase, allMetaQuery, imageSizes,
                       useCenterDists=True, usePointCoherence=True, useAngleCoherence=True, query_centroid_matrix=None,
                       returnClusters=True, visualize=False, IDToImage=None, d=None, visRank=100, visOutDir='.'):
    clusters_concat = np.concatenate(points_of_clusters).astype(int)
    all_vq_bin_ids = clusters_concat[:, 0].flatten()
    imagesForClusters = clusters_concat[:, 1].flatten()
    # print(points_of_clusters[:50])
    # imagesForClusters = IDs.flatten()[all_vq_bin_ids]
    # print('clength: ',len(points_of_clusters) )
    # print('clusters_concat: ',clusters_concat.shape)
    # print('unique ids:', np.unique(imagesForClusters).shape)
    qsize = getQuantizationSizes(imageSizes)
    # featureIDs_frombins = I.flatten()[all_vq_bin_ids]
    flat_voteCoords = voteCoords.reshape((-1, 2))
    # flat_metadata = allMetaDatabase.reshape((len(I.flatten()),-1))

    # imagesForClusters_contig = IDs_contig.flatten()[all_vq_bin_ids]
    if query_centroid_matrix is not None:
        mappedCentroids = query_centroid_matrix.reshape((-1, 2))[all_vq_bin_ids]
    mappedVotes = flat_voteCoords[all_vq_bin_ids]
    mappedMetaDB = allMetaDatabase.reshape((-1, allMetaDatabase.shape[2]))[all_vq_bin_ids]
    mappedMetaQ = allMetaQuery.reshape((-1, allMetaQuery.shape[2]))[all_vq_bin_ids]
    # mappedSizes = imageSize_matrix.reshape((-1,2))[all_vq_bin_ids]
    mappedMatchScores = Dinv.flatten()[all_vq_bin_ids]
    mappedVoteScores = mappedMatchScores
    mappedQSizes = qsize[all_vq_bin_ids]
    # after applying the imageID to each votecoord, we find the unique ordering (for bincount) and the unweighted counts of each cluster
    unique_vq_bin_ids, unique_vq_bin_inds, unique_vq_bin_ids_inv, unique_bins_count = np.unique(imagesForClusters,
                                                                                                return_inverse=True,
                                                                                                return_index=True,
                                                                                                return_counts=True)  # Cluster_centers_X[unique_vq_bin_ids_inv] will map cluster centers out to all points
    #     print(unique_vq_bin_ids_inv.shape)

    imagesForClusters_contig = np.arange(0, len(unique_vq_bin_ids))[unique_vq_bin_ids_inv]
    final_image_IDs = unique_vq_bin_ids
    # Cluster Center Vote
    if useCenterDists or usePointCoherence:
        #         print(imagesForClusters_contig.shape,np.unique(imagesForClusters_contig).shape,unique_bins_count.shape,mappedVotes.shape,np.bincount(imagesForClusters_contig).shape)
        Cluster_centers_X = np.bincount(imagesForClusters_contig, weights=mappedVotes[:, 0]) / unique_bins_count
        Cluster_centers_Y = np.bincount(imagesForClusters_contig, weights=mappedVotes[:, 1]) / unique_bins_count
        Cluster_centers_arr = np.vstack((Cluster_centers_X, Cluster_centers_Y)).T[unique_vq_bin_ids_inv]
    if useCenterDists:
        mappedPointDifs = mappedVotes - Cluster_centers_arr
        mappedPointDists = np.linalg.norm(mappedPointDifs, axis=1)
        mappedProbs = st.norm.pdf((mappedPointDists) / mappedQSizes, scale=1.5) / st.norm.pdf(0, scale=1.5)
        mappedVoteScores = mappedMatchScores * mappedProbs
    meandists = np.bincount(imagesForClusters_contig, weights=mappedPointDists) / unique_bins_count
    voteSums = np.bincount(imagesForClusters_contig, weights=mappedVoteScores)
    voteSums[voteSums == 0] = .00001
    voteScores = voteSums * np.log2(unique_bins_count)  # unique_bins_count*np.log2(voteSums)#
    # voteScores[voteScores < 0] = 0

    # Angle Coherence Score
    if useAngleCoherence:
        mappedAngleDifferences = mappedMetaDB[:, 3] - mappedMetaQ[:, 3]
        flipinds = mappedAngleDifferences < 0
        mappedAngleDifferences[flipinds] = (360 + mappedAngleDifferences[flipinds])
        # mappedAngleDifferences = (mappedAngleDifferences/8).astype(int)
        angleMeans = (np.bincount(imagesForClusters_contig, weights=mappedAngleDifferences) / unique_bins_count)[
            unique_vq_bin_ids_inv]
        angleSTDs = np.sqrt(np.bincount(imagesForClusters_contig, weights=np.square(
            mappedAngleDifferences - angleMeans)) / unique_bins_count) / unique_bins_count
        angleScore = 1 / (1 + angleSTDs)  # st.norm.pdf(angleSTDs,scale=2.5)/st.norm.pdf(0,scale=2.5)#
        voteScores *= angleScore
    # print(mappedAngleDifferences[imagesForClusters==goodid2])
    # print(angleSTDs[final_image_IDs == goodid2])
    # Point STD score
    if usePointCoherence:
        xSTDs = np.sqrt(
            np.bincount(imagesForClusters_contig, weights=np.square(mappedPointDifs[:, 0])) / unique_bins_count)
        ySTDs = np.sqrt(
            np.bincount(imagesForClusters_contig, weights=np.square(mappedPointDifs[:, 1])) / unique_bins_count)
        stdScores = 1 / (1 + np.sqrt(np.sqrt(np.sqrt((xSTDs + ySTDs)))) / 2)
        voteScores *= stdScores  # st.norm.pdf((xSTDs+ySTDs)/2,scale=2.5)/st.norm.pdf(0,scale=2.5)
        # print(xSTDs[final_image_IDs == goodid2],ySTDs[final_image_IDs == goodid2])
    # print(mappedVotes[imagesForCrlusters == goodid2])

    # visualizeVotes_toRank(flat_voteCoords,allMetaDatabase.reshape((-1,allMetaDatabase.shape[2]))[:,:2],invD.flatten(),mappedPointDists,mappedQSizes,imagesForClusters,mappedCentroids,final_image_IDs,voteScores,probename,rank=100,outdir = '/media/jbrogan4/scratch2/ICCV19/gld_surf/outvis')

    if visualize and IDToImage is not None and d is not None:
        visualizeVotes_toRank(mappedVotes, mappedMetaDB[:, :2], mappedMatchScores, mappedPointDists, mappedQSizes,
                              imagesForClusters, mappedCentroids, final_image_IDs, voteScores, probename, IDToImage, d,
                              rank=200, outdir=visOutDir)
    elif visualize and IDToImage is None:
        print('Error in visualizing scores: an IDToImage dictionary is required for iamge names')
    elif visualize and d is None:
        print(
            'Error in visualizing scores: the dictionary d is required for mapping image names to directory locations')
    # print(vmags.shape,mapped)
    if returnClusters:
        outcinds = imagesForClusters.argsort()
        outclusters = np.split(all_vq_bin_ids[outcinds], np.cumsum(unique_bins_count)[:-1])
        # print('num clusters:', len(outclusters), 'num bins: ',len(unique_bins_count),' num mapped dists: ',mappedPointDists.shape, 'num vqall: ', all_vq_bin_ids.shape, 'num inds: ',all_vq_bin_ids[outcinds].shape)
        return voteScores, final_image_IDs, unique_bins_count, outclusters, mappedPointDists, voteSums, meandists
    return voteScores, final_image_IDs, unique_bins_count


def visualizeVotes_toRank(mappedVotes, mappedPoints, mappedMatchScores, mappedDists, mappedQSizes, imagesForClusters,
                          query_centroids, imageIDs, imageScores, probename, IDToImage, d, rank=100, outdir=''):
    sortinds = imageScores.argsort()[::-1]
    sortedScores = imageScores[sortinds]
    sortedScores = (sortedScores - sortedScores.min()) / (sortedScores.max() - sortedScores.min())
    sortedIDs = imageIDs[sortinds]
    probeimg = cv2.imread(d[probename])
    outputdir = os.path.join(outdir, probename.split('.')[0])
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    print('visualizing...')
    bar = progressbar.ProgressBar()
    for r in bar(range(min(rank, len(sortedIDs)))):
        try:
            resultID = sortedIDs[r]
            resultname = IDToImage[str(resultID)]
            nscore = sortedScores[r]
            resultVis = visualizeVotes_forImage(mappedVotes, mappedPoints, mappedMatchScores, mappedDists, mappedQSizes,
                                                imagesForClusters, resultID, IDToImage, d, nscore=1)
            csize = len(mappedMatchScores[imagesForClusters == resultID])
            centroid = query_centroids[imagesForClusters == resultID][0]
            rad = min(15, int(max(probeimg.shape[0], probeimg.shape[1]) / 25))
            qimg_withCenter = cv2.circle(probeimg.copy(), (int(centroid[0]), int(centroid[1])), rad, (0, 0, 255), -1)
            savename = 'r' + str(r).zfill(3) + '_' + str(csize) + '_' + resultname
            qsavename = 'query_' + savename
            cv2.imwrite(os.path.join(outputdir, savename + '.png'), resultVis)
            cv2.imwrite(os.path.join(outputdir, qsavename + '.png'), qimg_withCenter)
            origdir = os.path.join(outputdir, 'originalImages')
            if not os.path.isdir(origdir):
                os.makedirs(origdir)
            cv2.imwrite(os.path.join(origdir, savename + '.png'), cv2.imread(d[resultname]))

        except:
            pass


def visualizeVotes_forImage(mappedVotes, mappedPoints, mappedMatchScores, mappedDists, mappedQSizes, imagesForClusters,
                            imageID, IDToImage, d, nscore=1):
    imageName = IDToImage[str(imageID)]
    imageDir = d[imageName]
    img = cv2.imread(imageDir)
    votemap = np.zeros(img.shape[:2], np.double)
    pdffunc = st.norm.pdf
    stepsPerVote = 20
    useInds = imagesForClusters == imageID
    votes = mappedVotes[useInds]
    voteScores = mappedMatchScores[useInds]
    matchDists = 1 / voteScores - 1
    matchDists = matchDists / matchDists.max()
    points = mappedPoints[useInds]
    voteDists = mappedDists[useInds]  # The radiuses
    qwindowSizes = mappedQSizes[useInds]
    windowsize = qwindowSizes[0]
    for i in range(len(votes)):
        radius = voteDists[i] * 2
        stepsize = 1
        xrng = np.arange(stepsize, radius, stepsize)
        pf = pdffunc(xrng / windowsize, scale=1.5) / pdffunc(0, scale=1.5)
        pf = (pf - pf.min()) / (1 - pf.min())
        linkern = np.concatenate((pf[::-1], [1], pf)).reshape((1, -1))
        kern = linkern.T * linkern
        # plt.imshow(kern)
        startx = int(votes[i][0] - kern.shape[1] / 2)
        starty = int(votes[i][1] - kern.shape[0] / 2)
        endx = startx + kern.shape[1]
        endy = starty + kern.shape[0]
        neededShape = votemap[starty:endy, startx:endx].shape
        votemap[starty:endy, startx:endx] += (kern * voteScores[i])[:neededShape[0], :neededShape[1]]

    cmap = plt.cm.jet
    # votemap = votemap.max()-votemap
    norm = plt.Normalize(vmin=votemap.min(), vmax=votemap.max())
    votemap *= nscore
    votemap = votemap.max() - votemap
    votemap = votemap - votemap.min()
    cmapimg = (cmap(np.square(norm(votemap)))[:, :, :-1] * 255).astype(np.uint8)
    overlay = cmapimg.copy()  # cv2.cvtColor(cmapimg.copy(),cv2.COLOR_BGR2RGB)
    output = img.copy()  # cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
    cv2.addWeighted(overlay, .5, img, .5, 0, output)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output
    plt.imshow(output)
    # vmags = np.linalg.norm(votes-points,axis=1)
    # print(vmags)
    # plt.scatter(voteDists,voteScores)

algorithmName = 'delf_1000_nh'
algorithmVersion = '2.0'
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

    def addScore(self, filename, score, ID=None, visData=None):
        self.scores[filename] = score
        if ID is not None:
            self.map[ID] = filename
        if visData is not None:
            self.visData[filename] = visData

    # this function merges two results
    def mergeScores(self, additionalScores, ignoreIDs=[]):
        # print('merging')
        for s in additionalScores.scores:
            if s in self.scores:
                if additionalScores.scores[s] > self.scores[s]:
                    self.scores[s] = additionalScores.scores[s]
            else:
                self.scores[s] = additionalScores.scores[s]
        if additionalScores.visData is not None:
            self.visData.update(additionalScores.visData)
            # sortinds = np.array(self.scores.values()).argsort()
            # vd = self.visData.copy()
            # self.visData.clear()
            # for v in np.array(list(vd.keys())).argsort()[::-1]:
            #   self.visData[v] = vd[v]
        sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores = sortedscores

    # this function merges two results
    def dictSort(self, additionalScores):
        od = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores.update(additionalScores.scores)
        sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores = sortedscores

    # Once scores are merged together, at most "numberOfResultsToRetrieve" will be retained
    def pairDownResults(self, numberOfResultsToRetrieve):
        numberOfResultsToRetrieve = int(numberOfResultsToRetrieve)
        if len(self.scores) > numberOfResultsToRetrieve:
            newscores = collections.OrderedDict(
                sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:numberOfResultsToRetrieve])
            self.scores = newscores

    def normalizeResults(self):
        maxVal = self.scores[list(self.scores.keys())[0]]
        for s in self.scores:
            self.scores[s] = self.scores[s] / maxVal


def createOutput(probeFilename, resultScores):
    return {'algorithm': createAlgOutput(), 'provenance_filtering': createFilteringOutput(probeFilename, resultScores)}


def createAlgOutput():
    return {'name': algorithmName.replace(" ", ""), 'version': algorithmVersion.replace(" ", "")}


def createFilteringOutput(probeFilename, resultScores):
    return {'probe': probeFilename, 'matches': resultScores.scores, 'meta': resultScores.visData}


def convertNISTJSON(results):
    jsonResults = {}
    nodes = []
    jsonResults['directed'] = True

    scores = results['provenance_filtering']['matches']
    meta = results['provenance_filtering']['meta']
    count = 1
    for filename in scores:
        node = {}
        node['id'] = str(count)
        node['file'] = 'world/' + filename
        node['fileid'] = os.path.splitext(os.path.basename(filename))[0]
        node['nodeConfidenceScore'] = scores[filename]
        # if meta is not None:
        #    node['meta'] = meta[filename]
        nodes.append(node)
        count = count + 1
    jsonResults['nodes'] = nodes
    jsonResults['links'] = []
    jsonstring = json.dumps(jsonResults)
    return jsonstring


def nhscore(I, D, image_level_IDs, imageSizes, meta_db, meta_query, visualize=False, visFolder='.', IDToImage=None,
            partitionDictionary=None,numberOfResultsToRetrieve=500):
    # checkIDs = np.array([ 384113  , 92602 ,1825963 , 685475, 1561460])
    # checkNames = np.array(['8509d920ca690537fcd7ba7b52826a1d','5906dc876f07d425667bb59656cf181c','3e1fbf6871cdc0ea5a926183859e5d27','b349d0e3b5e60fe79e12a4c1f36d46cb','1dc74e762b4372a9f3d90261d3312c94'])
    qsize = getQuantizationSizes(imageSizes, divis=4, power=.8)
    qsize_matrix = qsize.reshape((image_level_IDs.shape[0], -1, 1))
    invD = Dscore(D)
    voteCoords, IDs_contig, centroidMatrix = projectVotes(I, invD, image_level_IDs, meta_db, meta_query, imageSizes)
    points_of_clusters, size_of_clusters = runDensityClustering(voteCoords, qsize_matrix, image_level_IDs.flatten(),
                                                                IDs_contig, I.flatten(), removeMultiMatch=True,
                                                                removeMultiDatabase=True)
    voteScores, unique_vq_bin_ids, clusterSizes, clusters, pointDists, votesums, meandists = NHScore_Vectorized(
        voteCoords, points_of_clusters, invD, meta_db, meta_query, imageSizes, useCenterDists=True,
        usePointCoherence=False, useAngleCoherence=False, query_centroid_matrix=centroidMatrix, visualize=visualize,
        visOutDir=visFolder, visRank=200, IDToImage=IDToImage, d=partitionDictionary)
    sortinds = voteScores.argsort()[::-1]
    sortedScores = voteScores[sortinds]
    sortedIDs = unique_vq_bin_ids[sortinds]
    sortedSizes = clusterSizes[sortinds]
    sortedClusters = np.array(clusters)[sortinds]

    #     for id,name in zip(checkIDs,checkNames):
    #         r = np.where(sortedIDs == id)[0][0]
    #         print(name,'rank: ', r,' nkeypoints: ',sortedSizes[r])

    return sortedIDs,sortedScores,sortedScores.max()
    # r1 = filteringResults()
    # for i in range(0, min(len(sortedIDs), numberOfResultsToRetrieve * 10)):
    #     id = sortedIDs[i]
    #     id_str = str(int(id))
    #     if id_str in IDToImage:
    #         imname = IDToImage[id_str]
    #         score = sortedScores[i]
    #         r1.addScore(imname, score, ID=id)
    # return r1


def scoreFolder(folder, loaddir_orig, IDToImage, d,numberOfResultsToRetrieve):
    print('running ', folder)
    testbedPath = os.path.join(loaddir_orig, folder)
    outfileName = folder.split('.')[0] + '.json'
    outfilePath = os.path.join(outputFolder, outfileName)
    if os.path.exists(outfilePath):
        return
    try:
        if os.path.isdir(testbedPath):
            outfileName = folder.split('.')[0] + '.json'
            outfilePath = os.path.join(outputFolder, outfileName)
            print('loading files...')
            If = np.load(os.path.join(testbedPath, 'I.npy'))
            Df = np.load(os.path.join(testbedPath, 'D.npy'))
            allMetaDatabasef = np.load(os.path.join(testbedPath, 'metadb.npy'))
            allMetaQueryf = np.load(os.path.join(testbedPath, 'metaq.npy'))
            image_level_IDsf = np.load(os.path.join(testbedPath, 'image_level_IDs.npy')).astype(int)
            print(image_level_IDsf.shape)
            all_imageSizesf = np.load(os.path.join(testbedPath, 'imageSizes.npy'))
            imageSize_matrixf = all_imageSizesf.reshape(image_level_IDsf.shape[0], -1, 2)
            print('calc1')
            halfsize = max(1, int(If.shape[0] / 2))
            if If.shape[0] < 1:
                print('could not process ', os.path.basename, ' because it had no features')
                return
            allMetaDatabase = allMetaDatabasef[:halfsize]
            allMetaQuery = allMetaQueryf[:halfsize]
            image_level_IDs = image_level_IDsf[:halfsize]
            # all_imageSizes = all_imageSizes[:halfsize]
            imageSize_matrix = imageSize_matrixf[:halfsize]
            imageSizes = imageSize_matrix.reshape((-1, 2))
            I = If[:halfsize]
            D = Df[:halfsize]

            r1 = nhscore(I, D, image_level_IDs, imageSizes, allMetaDatabase, allMetaQuery, visualize=False,
                         visFolder='.', IDToImage=IDToImage, partitionDictionary=d)

            print('calc2')
            allMetaDatabase = allMetaDatabasef[halfsize:]
            allMetaQuery = allMetaQueryf[halfsize:]
            image_level_IDs = image_level_IDsf[halfsize:]
            # all_imageSizes = all_imageSizes[:halfsize]
            imageSize_matrix = imageSize_matrixf[halfsize:]
            imageSizes = imageSize_matrix.reshape((-1, 2))
            I = If[halfsize:]
            D = Df[halfsize:]
            r2 = nhscore(I, D, image_level_IDs, imageSizes, allMetaDatabase, allMetaQuery, visualize=False,
                         visFolder='.', IDToImage=IDToImage, partitionDictionary=d)

            r1.mergeScores(r2)
            r1.pairDownResults(numberOfResultsToRetrieve)
            finaloutput = createOutput(folder, r1)
            jout = convertNISTJSON(finaloutput)
            print('writing ', outfilePath)
            with open(outfilePath, 'w') as jsonFile:
                jsonFile.write(jout)
            return
    except:
        print('error')
        return

import sys
if __name__ == "__main__":
    inputFolderListPath = sys.argv[1]
    outputFolder = sys.argv[3]
    parentFolder = sys.argv[2]
    njobs = int(sys.argv[4])
    jobnum= int(sys.argv[5])-1
    with open(inputFolderListPath,'r') as fp:
        flistt = fp.readlines()
    flist = []
    for f in flistt:
        flist.append(f.rstrip())
    with open(os.path.join(parentFolder, '../imid_to_name.json'), 'r') as fp:
        IDToImage = json.load(fp)
    with open(os.path.join(parentFolder, 'worldDict.json'), 'r') as fp:
        d = json.load(fp)
    stride = len(flist)/njobs
    start =  int(stride*jobnum)
    end = min(len(flist),int(stride*(jobnum+1)))
    folders = flist[start:end]
    print('running ', len(folders),'folders')
    for f in folders:
        scoreFolder(f,parentFolder,IDToImage,d,500)
        print('ran ',f)