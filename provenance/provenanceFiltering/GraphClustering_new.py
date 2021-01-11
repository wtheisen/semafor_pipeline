import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering,SpectralClustering,spectral_clustering
import progressbar
from scipy.sparse import csr_matrix,lil_matrix,save_npz,load_npz
import collections
import argparse
from fileutil import make_sure_path_exists
import skfuzzy
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import glob
import markov_clustering as mcl

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


def createOutput(probeFilename, resultScores, algorithmName, algorithmVersion):
    return {'algorithm': createAlgOutput(algorithmName, algorithmVersion), 'provenance_filtering': createFilteringOutput(probeFilename, resultScores)}


def createAlgOutput(algorithmName, algorithmVersion):
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

def do_markov_clustering(totalMat, saveFolder):
    nxmat = nx.from_scipy_sparse_matrix(totalMat)
    adj_matrix = nx.to_numpy_matrix(nxmat)
    res = mcl.run_mcl(adj_matrix)
    clusterID_list = mcl.get_clusters(res)

    j = 0
    algorithmVersion = '1'
    algorithmName = "Markov"
    for clusterID in clusterID_list:
        clusterID = list(clusterID)
        print('Generating results for cluster ', j)
        # print(clusters.labels_)
        ims_for_cluster = np.array(allImagePaths)[clusterID]
        # ims_for_cluster = [allImagePaths[x] for x in clusters.labels_ if x == clusterID ]
        # print(ims_for_cluster)
        r1 = filteringResults()
        bar = progressbar.ProgressBar()
        for i in bar(range(0, len(ims_for_cluster))):
            impath = ims_for_cluster[i]
            imname = os.path.basename(impath)
            score = 1
            r1.addScore(imname, score, ID=0)
        savename = 'cluster' + str(j).zfill(3) + '_size' + str(len(ims_for_cluster)) + '_algorithm_' + algorithmName + '.json'
        # print(r1)
        jout = createOutput(savename, r1, algorithmName, algorithmVersion)
        jout = convertNISTJSON(jout)
        j = j + 1

        with open(os.path.join(saveFolder, savename), 'w') as fp:
            fp.write(jout)

def do_spectral_clustering(totalMat, cacheFolder, saveFolder):
    sparseMat = totalMat.tocsr()
    print('Shape: ', sparseMat.shape)
    print('clustering...')
    amOutPath = os.path.join(cacheFolder, 'affinityMatrix')
    save_npz(amOutPath, sparseMat)

    labels = spectral_clustering(sparseMat, n_clusters=numClusters, eigen_solver='arpack', random_state=15)

    # labels = spectral_clustering(sparseMat,n_clusters=100,eigen_solver='arpack')
    # clusterfunc =SpectralClustering(assign_labels="discretize",random_state=0,eigen_solver='arpack',affinity='precomputed_nearest_neighbors',n_clusters=100)
    # clusters = clusterfunc.fit(sparseMat)
    # u,counts=np.unique(clusters.labels_,return_counts=True)
    u, counts = np.unique(labels, return_counts=True)

    algorithmVersion = '1'
    algorithmName = 'SpectralClustering_200'
    for clusterID in u:
        print('Generating results for cluster ', clusterID)
        # print(clusters.labels_)
        ims_for_cluster = np.array(allImagePaths)[labels == clusterID]
        # ims_for_cluster = [allImagePaths[x] for x in clusters.labels_ if x == clusterID ]
        # print(ims_for_cluster)
        r1 = filteringResults()
        bar = progressbar.ProgressBar()
        for i in bar(range(0, len(ims_for_cluster))):
            impath = ims_for_cluster[i]
            imname = os.path.basename(impath)
            score = 1
            r1.addScore(imname, score, ID=0)
        savename = 'cluster' + str(clusterID).zfill(3) + '_size' + str(len(ims_for_cluster)) + '_algorithm_' + algorithmName + '.json'
        # print(r1)
        jout = createOutput(savename, r1, algorithmName, algorithmVersion)
        jout = convertNISTJSON(jout)

        with open(os.path.join(saveFolder, savename), 'w') as fp:
            fp.write(jout)


parser = argparse.ArgumentParser()
parser.add_argument('--FilteringResultFolder', help='folder containing filtering result jsons')
parser.add_argument('--ImageRoot', help='Root Image Directory')
parser.add_argument('--CacheFolder', help='where to save all intermediate files',default='.')
parser.add_argument('--ImageFileList', help='list of image files')
parser.add_argument('--OutputFolder', help='output folder')
parser.add_argument('--Algorithm', help='Markov or Spectral')
parser.add_argument('--k', help='number of clusters')

args = parser.parse_args()

resultsFolder = args.FilteringResultFolder
dataFolder = args.ImageRoot
cacheFolder = args.CacheFolder
allImageFilesPath = args.ImageFileList
saveFolder = args.OutputFolder
Algorithm = args.Algorithm

numClusters = int(args.k)
print(f'Clustering with k = {numClusters}...')

make_sure_path_exists(saveFolder)
make_sure_path_exists(cacheFolder)

files = glob.glob(saveFolder + "/*")
for f in files:
    os.remove(f)


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
# totalMat = np.memmap(os.path.join(cacheFolder,'affinityMatrix'),dtype='float32',mode='w+',shape=(len(allImageNames),len(allImageNames)))

totalMat = lil_matrix((len(allImageNames),len(allImageNames)),dtype='float32')

print('building affinity matrix...')
for r in bar(allJsons):
    # print('JSON PATH: ', r)
    resultID = os.path.splitext(os.path.basename(r))[0]
    try:
        edgeStartID = imageIDtoIndex[resultID]
    except:
        continue
    with open(r,'r') as fp:
        d = json.load(fp)
    nodes = d['nodes']
    row = imageIDtoIndex[resultID]
    for n in nodes:
        nid = n['fileid']
        column = imageIDtoIndex[(nid)]
        weight = float(n['nodeConfidenceScore'])
        edgeEndID = imageIDtoIndex[nid]
        totalMat[row,column] = weight
        edgeList.append(str(edgeStartID) + ' ' + str(edgeEndID) + ' ' + str(weight))
        mappedNodes.append(nid)
        edgeWeights.append(weight)

if Algorithm == "Markov":
    do_markov_clustering(totalMat, saveFolder)

if Algorithm == 'Spectral':
    do_spectral_clustering(totalMat, cacheFolder, saveFolder)
