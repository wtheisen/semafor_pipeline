import os
import sys
import numpy as np


def findBestDenseCluster(points, maxDistance):
    pointsSortedByXIndexes = np.argsort(points[:, 0])
    pointsSortedByYIndexes = np.argsort(points[:, 1])
    pointsSortedByX = points[pointsSortedByXIndexes]
    pointsSortedByY = points[pointsSortedByYIndexes]
    pointsSortedByXSquared = np.square(pointsSortedByX)
    pointsSortedByYSquared = np.square(pointsSortedByY)
    xSortedByX = points[:, 0][pointsSortedByXIndexes]
    ySortedByY = points[:, 1][pointsSortedByYIndexes]
    validClosestPoints = []
    maxDistanceSquared = maxDistance * maxDistance
    for i in range(0, points.shape[0]):
        p = points[i]
        closestPoints = []
        range_x = np.searchsorted(xSortedByX, [p[0] - maxDistance, p[0] + maxDistance])
        range_y = np.searchsorted(ySortedByY, [p[0] - maxDistance, p[0] + maxDistance])
        range_x[1] = max(range_x[1] - 1, 0)
        range_y[1] = max(range_y[1] - 1, 0)

        yFromValidXSquared = pointsSortedByXSquared[range_x[0]:range_x[1], 1]
        maxYsquared = maxDistanceSquared - pointsSortedByXSquared[:, 1][range_x[0]:range_x[1]]
        validIndexesFromX = pointsSortedByXIndexes[range_x[0]:range_x[1]][(yFromValidXSquared - maxYsquared) >= 0]

        xFromValidYSquared = pointsSortedByYSquared[range_y[0]:range_y[1], 0]
        maxXsquared = maxDistanceSquared - pointsSortedByYSquared[:, 0][range_y[0]:range_y[1]]
        validIndexesFromY = pointsSortedByYIndexes[range_y[0]:range_y[1]][(xFromValidYSquared - maxXsquared) >= 0]
        validPointIndexes = np.unique(np.concatenate(validIndexesFromX, validIndexesFromY))

def findBestDenseClustersForTensor(voteCoords,maxDistance):
    pass

def findBestDenseClusterUsingTable(points, maxDistance):

    # # initial simple vector quantization
    vq = (points / maxDistance).astype(int)
    vq_shiftedRight = ((points-np.array([maxDistance/2,0]))/maxDistance).astype(int)
    vq_shiftedDown = ((points-np.array([0,maxDistance/2]))/maxDistance).astype(int)
    maxDensity = -1000000000
    vq_ids_max = None
    MaxDensityBinID = None
    for vqAll in [vq,vq_shiftedRight,vq_shiftedDown]:
        minXbin = vqAll[:, 0].min()
        minYbin = vqAll[:, 1].min()
        maxXbin = vqAll[:, 0].max() - minXbin
        maxYbin = vqAll[:, 1].max() - minYbin
        vq_norm = vqAll - np.array([minXbin, minYbin])

        # Find bin (with neighbors)
        vq_ids = np.ravel_multi_index(vq_norm.T, (maxXbin+1, maxYbin+1))
        uniqueIDS, binCount = np.unique(vq_ids, return_counts=True)
        MaxDensityBinIndex = np.argmax(binCount)
        MaxDensityBinID_candidate = uniqueIDS[MaxDensityBinIndex]
        binDensity = binCount[MaxDensityBinIndex]
        if binDensity > maxDensity:
            maxDensity = binDensity
            vq_ids_max = vq_ids
            MaxDensityBinID = MaxDensityBinID_candidate
        #maxDensityBinCoord = np.unravel_index(np.asarray([MaxDensityBinID]),(maxXbin+1, maxYbin+1))

    allBestPointIndexes = np.arange(0, len(vq_ids_max))[vq_ids_max == MaxDensityBinID]
    bestPoints = points[allBestPointIndexes]
    return [1],bestPoints,np.ones(bestPoints.shape[0]),allBestPointIndexes