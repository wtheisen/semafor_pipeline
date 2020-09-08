import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering,SpectralClustering
import progressbar
from scipy.sparse import csr_matrix,save_npz,load_npz
import collections
import argparse
from fileutil import make_sure_path_exists

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
        # self.scores.update(additionalScores.scores)
        for s in additionalScores.scores:
            if s in self.scores:
                if additionalScores.scores[s] > self.scores[s]:
                    self.scores[s] = additionalScores.scores[s]
            else:
                self.scores[s] = additionalScores.scores[s]

        newscores = collections.OrderedDict()
        vals = np.array(list(self.scores.values()))
        #print('vals shape: ', vals.shape)
        keys = np.array(list(self.scores.keys()))
        #print('keys shape: ', keys.shape)
        resort = vals.argsort()[::-1]
        for i in resort:
            newscores[keys[i]] = vals[i]
        self.scores = newscores

    #         if additionalScores.visData is not None:
    #             self.visData.update(additionalScores.visData)
    #             #sortinds = np.array(self.scores.values()).argsort()
    #             #vd = self.visData.copy()
    #             #self.visData.clear()
    #             #for v in np.array(list(vd.keys())).argsort()[::-1]:
    #             #   self.visData[v] = vd[v]
    #         sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
    #         self.scores = sortedscores

    # this function merges two results
    def dictSort(self, additionalScores):
        od = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores.update(additionalScores.scores)
        sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores = sortedscores

    # Once scores are merged together, at most "numberOfResultsToRetrieve" will be retained
    def pairDownResults(self, numberOfResultsToRetrieve):
        numberOfResultsToRetrieve = int(numberOfResultsToRetrieve)
        newscores = collections.OrderedDict()
        vals = np.array(list(self.scores.values()))
        print('vals shape: ', vals.shape)
        keys = np.array(list(self.scores.keys()))
        print('keys shape: ', keys.shape)
        resort = vals.argsort()[::-1]
        for i in resort[:numberOfResultsToRetrieve]:
            newscores[keys[i]] = vals[i]
        self.scores = newscores

    #         if len(self.scores) > numberOfResultsToRetrieve:
    #             newscores = collections.OrderedDict(
    #                 sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:numberOfResultsToRetrieve])
    #             self.scores = newscores
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
    #print(jsonstring)
    return jsonstring


parser = argparse.ArgumentParser()
parser.add_argument('--FilteringResultFolder', help='folder containing filtering result jsons')
parser.add_argument('--ImageRoot', help='Root Image Directory')
parser.add_argument('--CacheFolder', help='where to save all intermediate files',default='.')
parser.add_argument('--ImageFileList', help='list of image files')
parser.add_argument('--OutputFolder', help='output folder')

args = parser.parse_args()

resultsFolder = args.FilteringResultFolder
dataFolder = args.ImageRoot
cacheFolder = args.CacheFolder
allImageFilesPath = args.ImageFileList
saveFolder = args.OutputFolder

make_sure_path_exists(saveFolder)
make_sure_path_exists(cacheFolder)


with open(allImageFilesPath,'r') as fp:
    allImagePaths = [l.rstrip() for l in fp.readlines()]
allJsons = [os.path.join(resultsFolder,d) for d in os.listdir(resultsFolder)]
imageDict = {}
allImageNames = []
allImageIDs = []
imageIDtoIndex = {}
count = 0
for im in allImagePaths:
    imname = os.path.basename(im)
    imid = os.path.splitext(imname)[0]
    imageDict[imname] = im
    allImageNames.append(imname)
    allImageIDs.append(imid)
    imageIDtoIndex[imid] = count
    count += 1

bar = progressbar.ProgressBar()
edgeList = []
mappedNodes = []
edgeWeights = []

# totalMat = np.zeros((len(allImageNames),len(allImageNames)),dtype=np.float32)
print('reading affinity matrix from ',os.path.join(cacheFolder,'affinityMatrix'))
totalMat = np.memmap(os.path.join(cacheFolder,'affinityMatrix'),dtype='float32',mode='w+',shape=(len(allImageNames),len(allImageNames)))
print('building affinity matrix...')
for r in bar(allJsons):
    resultID = os.path.splitext(os.path.basename(r))[0]
    edgeStartID = imageIDtoIndex[resultID]
    with open(r,'r') as fp:
        d = json.load(fp)
    nodes = d['nodes']
    row = imageIDtoIndex[resultID]
    print('nodes: ',len(nodes))
    for n in nodes:
        nid = n['fileid']
        column = imageIDtoIndex[(nid)]
        weight = n['nodeConfidenceScore']
        edgeEndID = imageIDtoIndex[nid]
        totalMat[row,column] = weight
        edgeList.append(str(edgeStartID) + ' ' + str(edgeEndID) + ' ' + str(weight))
        mappedNodes.append(nid)
        edgeWeights.append(weight)
sparseMat = csr_matrix(totalMat)
print('clustering...')
clusterfunc =SpectralClustering(assign_labels="discretize",random_state=0,eigen_solver='arpack',affinity='precomputed_nearest_neighbors',n_clusters=150)
clusters = clusterfunc.fit(sparseMat)
u,counts=np.unique(clusters.labels_,return_counts=True)

algorithmVersion = '1'
algorithmName = 'SpectralClustering_200'
for clusterID in u:
    print('Generating results for cluster ', clusterID)
    #print(clusters.labels_)
    ims_for_cluster = np.array(allImagePaths)[clusters.labels_ == clusterID]
    #ims_for_cluster = [allImagePaths[x] for x in clusters.labels_ if x == clusterID ]
    # print(ims_for_cluster)
    r1 = filteringResults()
    bar = progressbar.ProgressBar()
    for i in bar(range(0, len(ims_for_cluster))):
        impath = ims_for_cluster[i]
        imname = os.path.basename(impath)
        score = 1
        r1.addScore(imname, score, ID=0)
    savename = '4chan_cluster' + str(clusterID).zfill(3) + '_size' + str(len(ims_for_cluster)) + '.json'
    #print(r1)
    jout = createOutput(savename, r1)
    jout = convertNISTJSON(jout)

    with open(os.path.join(saveFolder, savename), 'w') as fp:
        fp.write(jout)
