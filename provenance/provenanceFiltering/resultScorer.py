import os
import sys
import json
import numpy as np
import progressbar
import argparse

worldIndexList = []
worldIndexExtMap = {}
nodeMap = {}
nodeMap_reverse = {}
class GraphData:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
def getNodeID(fname,gtJournal):
    if fname in nodeMap_reverse:
        fname = nodeMap_reverse[fname]
    theID = -1
    c = 0
    for n in gtJournal['nodes']:
        if n['file'] == fname or n['id'] == fname:
            theID = c
            break
        c+=1
    return theID
def getMissingNodesFromExample(resultJson,gtJournal,world_index_list,rank=-1):
    foundNodes = resultJson['nodes']
    if rank > 0:
        foundNodes = foundNodes[:min(len(foundNodes),rank)]
    trueNodes = gtJournal['nodes']
    foundNodeNames =[]
    trueNodeNames = []

    correctNodeNames = []
    missingNodeNames = []
    erroneousNodeNames = []
    unusedNodeNames = []

    missingNodeIndexes = []
    correctNodeIndexes = []
    erroneousNodeIndexes = []
    nodesNotInWorldIndexes = []
    for n in trueNodes:
        trueNodeNames.append(os.path.basename(n['file']))
    for n in foundNodes:
        foundNodeNames.append(os.path.basename(n['file']))
    c = 0
    for fileName in trueNodeNames:
        if fileName in world_index_list:
            if fileName in foundNodeNames:
                correctNodeNames.append(fileName)
                correctNodeIndexes.append(c)
            elif fileName not in foundNodeNames:
                missingNodeNames.append(fileName)
                missingNodeIndexes.append(c)
        else:
            unusedNodeNames.append(fileName)
        c+=1
    c = 0
    for fileName in foundNodeNames:
        if fileName in world_index_list:
            if fileName not in trueNodeNames:
                erroneousNodeNames.append(fileName)
        c+=1
    return missingNodeIndexes,correctNodeIndexes,erroneousNodeNames,unusedNodeNames

def buildAdjListFromJson(jsonGraph):
    graph_down = {}
    graph_up = {}
    graph_undirected = {}
    roots = []
    leafs = []
    amatrix = np.zeros((len(jsonGraph['nodes']),len(jsonGraph['nodes'])))
    linkOps = {}
    for l in jsonGraph['links']:
        amatrix[l['source'],l['target']] = 1
        linkOps[(l['source'],l['target'])] = l['op']
    for i in range(len(jsonGraph['nodes'])):
        children = []
        parents = []
        for l in jsonGraph['links']:
            if l['source'] == i: children.append(l['target'])
            if l['target'] == i: parents.append(l['source'])
        if len(children) == 0:
            leafs.append(i)
        if len(parents) == 0:
            roots.append(i)
        graph_down[i] =  set(children)
        graph_up[i] = set(parents)
        both = set(children)
        both.update(set(parents))
        graph_undirected[i] = both

    r = GraphData(alist_down=graph_down,alist_up=graph_up,graph_undirected=graph_undirected,amatrix=amatrix,roots=roots,leafs=leafs,json=jsonGraph,linkOps=linkOps)
    return r

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for next in set(graph[start]) - visited:
        dfs(graph, next, visited)
    return visited

def dfs_paths(graph, start, goal, path=None):
    if path is None:
        path = [start]
    if start == goal:
        yield path
    for next in set(graph[start]) - set(path):
        yield from dfs_paths(graph, next, goal, path + [next])

def dfs_toLeafs(graph, start, path=None):
    if path is None:
        path = [start]
    if len(graph[start]) == 0:
        yield path
    for next in set(graph[start]) - set(path):
        yield from dfs_toLeafs(graph, next, path + [next])

def getChainRankOfMissingNodes(missingNodeIndexes,gtJournal):
    graph_down, graph_up = buildAdjListFromJson(gtJournal)
    for i in missingNodeIndexes:
        node = gtJournal['nodes'][i]
        visited = dfs(graph_down,i)
def remapNodes(jsonfile,nodeMap):
    for n in jsonfile['nodes']:
        filename = n['file']
        folder = os.path.dirname(filename)
        filename = os.path.basename(filename)
        fparts = filename.split('.')
        fileID = fparts[0]
        fileExt = fparts[1]
        if fileID in nodeMap:
            newfileID = nodeMap[fileID]
            n['file']=os.path.join(folder,newfileID)

def determineMinimumPenalty(nodeID,graphItem):
    totalPenalty = 0
    parents = []
    children = []
    if nodeID in graphItem.alist_up:
        parents = graphItem.alist_up[nodeID]
    if nodeID in graphItem.alist_down:
        children = graphItem.alist_down[nodeID]
    totalPenalty += len(parents) #Penalty for not connecting nodeID to its parents
    if nodeID in graphItem.alist_down[nodeID]:
        totalPenalty += len(graphItem.alist_down[nodeID])*2 #Penalty for both not connecting the children to NodeID, and connecting them to something higher up
    return totalPenalty

def analyzeNodeInGraph(nodeName,probeName,graphItem):
    nodeID = getNodeID(nodeName,graphItem.json)
    probeID = getNodeID(probeName,graphItem.json)
    pathsToLeaves = list(dfs_toLeafs(graphItem.alist_down, nodeID))
    pathsToRoots = list(dfs_toLeafs(graphItem.alist_up,nodeID))
    pathsToProbe = list(dfs_paths(graphItem.graph_undirected,nodeID,probeID))
    minPathToLeaf = None
    minPathToRoot = None
    minPathToProbe = None
    allNodes_above = []
    allNodes_below = []
    minLength = 100000000000
    for p in pathsToLeaves:
        if len(p) < minLength:
            minLength = len(p)
            minPathToLeaf = p
        allNodes_below.extend(p[1:])
    allNodes_below = list(set(allNodes_below))
    minLength = 100000000000
    for p in pathsToRoots:
        if len(p) < minLength:
            minLength = len(p)
            minPathToRoot = p
        allNodes_above.extend(p[1:])
    allNodes_above = list(set(allNodes_above))
    minLength = 100000000000
    minBackwards = 10000000000
    for p in pathsToProbe:
        totalBackwards = 0
        for j in range(len(p)-1):
            i1 = p[j]
            i2 = p[j+1]
            if not graphItem.amatrix[i1,i2] == 1:
                totalBackwards += 1
        if totalBackwards < minBackwards:
            minBackwards = totalBackwards
            minLength = len(p)
            minPathToProbe = p
        elif totalBackwards == minBackwards:
            if len(p) < minLength:
                minLength = len(p)
                minPathToProbe = p
    shortestUndirectedPathLength = len(minPathToProbe)
    totalChain = minPathToRoot[::-1]
    totalChain.extend(minPathToLeaf[1:])
    totalChain = totalChain
    minPathToLeaf_names = []
    minPathToRoot_names = []
    minPathToProbe_names = []
    totalChain_names = []

    for i in minPathToLeaf:
        fname = graphItem.json['nodes'][i]['file']
        if fname in nodeMap_reverse:
            fname = nodeMap_reverse[fname]
        minPathToLeaf_names.append(fname)
    for i in minPathToRoot:
        fname = graphItem.json['nodes'][i]['file']
        if fname in nodeMap_reverse:
            fname = nodeMap_reverse[fname]
        minPathToRoot_names.append(fname)
    for i in minPathToProbe:
        fname = graphItem.json['nodes'][i]['file']
        if fname in nodeMap_reverse:
            fname = nodeMap_reverse[fname]
        minPathToProbe_names.append(fname)
    for i in totalChain:
        fname = graphItem.json['nodes'][i]['file']
        if fname in nodeMap_reverse:
            fname = nodeMap_reverse[fname]
        totalChain_names.append(fname)
    pathToProbeOps = []
    pathToLeafOps = []
    for j in range(len(minPathToProbe)-1):
        i1 = minPathToProbe[j]
        i2 = minPathToProbe[j+1]
        if (i1,i2) in graphItem.linkOps:
            op = graphItem.linkOps[(i1,i2)]
        else:
            op = graphItem.linkOps[(i2, i1)]
        pathToProbeOps.append(op)
    for j in range(len(minPathToLeaf)-1):
        i1 = minPathToLeaf[j]
        i2 = minPathToLeaf[j+1]
        if (i1,i2) in graphItem.linkOps:
            op = graphItem.linkOps[(i1,i2)]
        else:
            op = graphItem.linkOps[(i2, i1)]
        pathToLeafOps.append(op)
    # Chain degree separation
    prevVal = 1
    chainJumps = 0
    otherdist = 0
    for j in range(len(totalChain)-1):
        i1 = totalChain[j]
        i2 = totalChain[j+1]
        val = graphItem.amatrix[i1,i2]
        if not val == prevVal and val == 0:
            chainJumps+=1
        elif val == prevVal:
            otherdist+=1
        prevVal = val

    isLeaf = False
    if nodeID in graphItem.leafs:
        isLeaf = True
    isRoot = False
    if nodeID in graphItem.roots:
        isRoot = True
    isAboveProbe = False
    isBelowProbe = False
    if probeID in allNodes_below:
        isAboveProbe = True
    elif probeID in allNodes_above:
        isBelowProbe = True

    isInProbeChain = False
    if isAboveProbe or isBelowProbe:
        isInProbeChain = True
    isDonorToProbe = False
    isReceiverOfProbe = False
    if isAboveProbe and 'Donor' in pathToProbeOps:
        isDonorToProbe = True
    if isBelowProbe and 'Donor' in pathToProbeOps:
        isReceiverOfProbe = True
    isDonorAtSomePoint = False
    if 'Donor' in pathToLeafOps:
        isDonorAtSomePoint = True
    distanceFromRoot = len(minPathToRoot)
    distanceFromLeaf = len(minPathToLeaf)
    dfr_normalized = len(minPathToRoot)/len(totalChain)
    dfl_normalized = len(minPathToLeaf)/len(totalChain)
    distanceFromProbe_undirected = len(minPathToProbe)
    distanceFromProbe_withChainJumps = float(str(chainJumps) + '.' + str(otherdist))
    degree_in = len(graphItem.alist_up[nodeID])
    degree_out = len(graphItem.alist_down[nodeID])
    minimumPenalty = determineMinimumPenalty(nodeID,graphItem)
    return [degree_in,degree_out,isLeaf,isRoot,isInProbeChain,isAboveProbe,isBelowProbe,isDonorToProbe,isReceiverOfProbe,isDonorAtSomePoint,distanceFromLeaf,distanceFromRoot,dfr_normalized,dfl_normalized,distanceFromProbe_undirected,distanceFromProbe_withChainJumps,chainJumps,minimumPenalty]

def runGraphAnalysis(journalPath,jsonPath,worldIndexList,worldIndexExtMap,nodeMap,nodeMap_reverse,recalls=[100]):
    with open(journalPath, 'r') as fp:
        gtJournal = json.load(fp)
    with open(jsonPath, 'r') as fp:
        resultJson = json.load(fp)
    # allNodeIDs = []
    # for n in gtJournal['nodes']:
    #     allNodeIDs.append(n['id'])
    # uniqueIDs, counts = np.unique(np.asarray(allNodeIDs),return_counts=True)
    # print('duplicates: ', len(allNodeIDs)-len(uniqueIDs))
    remapNodes(gtJournal, nodeMap)
    recallDict = {}
    for recallNum in recalls:
        missingNodeIndexes, correctNodeIndexes, erroneousNodeIndexes, unused = getMissingNodesFromExample(resultJson, gtJournal,
                                                                                                  worldIndexList,recallNum)

        graphItem = buildAdjListFromJson(gtJournal)
        probeName = nodeMap_reverse[worldIndexExtMap[os.path.basename(jsonPath).split('.')[0]]]

        headerLine = 'probeName,nodeName,isMissing,degree_in,degree_out,isLeaf,isRoot,isInProbeChain,isAboveProbe,isBelowProbe,isDonorToProbe,isReceiverOfProbe,isDonorAtSomePoint,distanceFromLeaf,distanceFromRoot,dfr_normalized,dfl_normalized,distanceFromProbe_undirected,distanceFromProbe_withChainJumps,chainJumps,minimumPenalty'
        csvOutput = []
        correctDonorNodes = []
        correctMainNodes = []
        missedDonorNodes = []
        missedMainNodes = []
        missedDonorAboveNodes = []
        correctDonorAboveNodes = []
        missedAllNodes = []
        correctAllNodes = []
        for i in missingNodeIndexes:
            fname = gtJournal['nodes'][i]['file']
            if fname in worldIndexList:
                if fname in nodeMap_reverse:
                    fname_r = nodeMap_reverse[fname]
                    prefix_data = [os.path.basename(jsonPath),fname,True]
                    allData = analyzeNodeInGraph(fname_r, probeName, graphItem)
                    isDonorToProbe = allData[7]
                    isDonorAtAll = allData[9]
                    isCompositMadeOfProbe = allData[8]
                    if isDonorAtAll:
                        missedDonorNodes.append(fname)
                        if isDonorToProbe:
                            missedDonorAboveNodes.append(fname)
                    else:
                        missedMainNodes.append(fname)
                    missedAllNodes.append(fname)
        for i in correctNodeIndexes:
            fname = gtJournal['nodes'][i]['file']
            if fname in worldIndexList:
                if fname in nodeMap_reverse:
                    fname_r = nodeMap_reverse[fname]
                    prefix_data = [os.path.basename(jsonPath), fname, False]
                    allData = analyzeNodeInGraph(fname_r, probeName, graphItem)
                    isDonorToProbe = allData[7]
                    isDonorAtAll = allData[9]
                    isCompositMadeOfProbe = allData[8]
                    if isDonorAtAll:
                        correctDonorNodes.append(fname)
                        if isDonorToProbe:
                            missedDonorAboveNodes.append(fname)
                    else:
                        correctMainNodes.append(fname)
                    correctAllNodes.append(fname)
        recallDict[recallNum] = (correctAllNodes,missedAllNodes,correctMainNodes,missedMainNodes,correctDonorNodes,missedDonorNodes,correctDonorAboveNodes,missedDonorAboveNodes)


    return recallDict


def runGraphSet(journalDir,jsonDir,worldIndexPath,nodeRefPath,journalRefPath):
    probeToJournalMap = {}
    with open(worldIndexPath, 'r') as fp:
        lines = fp.readlines()
    for l in lines[1:]:
        parts = l.split('|')
        if len(parts) > 1:
            fname = os.path.basename(parts[2])
            fname_noext = fname.split('.')[0]
            worldIndexList.append(fname)
            worldIndexExtMap[fname_noext] = fname
    with open(nodeRefPath, 'r') as fp:
        lines = fp.readlines()
    for l in lines:
        l = l.rstrip()
        parts = l.split('|')
        if len(parts) > 1:
            nodeMap[parts[3]] = os.path.basename(parts[2])
            nodeMap_reverse[os.path.basename(parts[2])] = parts[3]

    with open(journalRefPath,'r') as fp:
        lines = fp.readlines()
    for l in lines[1:]:
        l = l.rstrip()
        parts = l.split('|')
        if len(parts) > 1:
            journalName = os.path.basename(parts[6])
            probeName = parts[1]
            probeToJournalMap[probeName] = journalName

    recalls = [25,50,100,200,300,400]
    bar = progressbar.ProgressBar()
    allNodes = []
    for jsonGraphFile in os.listdir(jsonDir):
        jsonFile = os.path.join(jsonDir,jsonGraphFile)
        probeName = jsonGraphFile.split('.')[0]
        if probeName in probeToJournalMap:
            journalName = probeToJournalMap[probeName]
            journalPath = os.path.join(journalDir,journalName)
            allData = runGraphAnalysis(journalPath,jsonFile,worldIndexList,worldIndexExtMap,nodeMap,nodeMap_reverse,recalls=recalls)
            highestRanks = allData[recalls[-1]]
            allRecallDenom = max(1,(len(highestRanks[0])+len(highestRanks[1])))
            allMainRecallDenom = max(1,(len(highestRanks[2]) + len(highestRanks[3])))
            allDonorRecallDenom = max(1,(len(highestRanks[4]) + len(highestRanks[5])))
            higherDonorRecallDenom = max(1,(len(highestRanks[6]) + len(highestRanks[7])))

            allRecall = len(highestRanks[0])/allRecallDenom
            allMainRecall = len(highestRanks[2]) / allMainRecallDenom
            allDonorRecall = len(highestRanks[4]) / allDonorRecallDenom
            higherDonorRecall = len(highestRanks[6]) / higherDonorRecallDenom
            # print(probeName, ' | ', journalName)
            # print('total recall: ',allRecall,' host recall: ', allMainRecall, ' all donor recall: ', allDonorRecall, ' higher donor recall: ', higherDonorRecall)
            # print('host recall: ', allMainRecall, ' all donor recall: ', allDonorRecall)

            allNodes.append(allData)
    finalResultsForRank = {}
    for rank in recalls:
        totalFound = 0
        totalMissed = 0
        donorsFound = 0
        donorsMissed = 0
        donorsAboveFound = 0
        donorsAboveMissed = 0
        hostsFound = 0
        hostsMissed = 0

        for result in allNodes:
            stForRank = result[rank]
            totalFound += len(stForRank[0])
            totalMissed += len(stForRank[1])
            hostsFound += len(stForRank[2])
            hostsMissed += len(stForRank[3])
            donorsFound += len(stForRank[4])
            donorsMissed += len(stForRank[5])
            donorsAboveFound += len(stForRank[6])
            donorsAboveMissed += len(stForRank[7])
        print('Total Recall@', rank, ': ',totalFound/max(totalFound+totalMissed,1))
        print('Donor Recall@',rank,': ',donorsFound/max(donorsFound+donorsMissed,1))
    # print('writing ', len(totalCSV),' lines...')
    # with open('allOutput.csv','w') as fp:
    #     fp.write('\n'.join(totalCSV))


# journalPath = '/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/reference/provenance/journals/88f18d781fdebd6836ac44fcaa656c1b.json'
# jsonPath = '/home/jbrogan4/Desktop/NC2017_results/e68047950999e492e8a024958e54a83a.json'
# worldIndexPath = '/Users/joel/Documents/Projects/Medifor/results/indexes/NC2017_Dev1-provenancefiltering-world.csv'
# nodeRefPath = '/Users/joel/Documents/Projects/Medifor/results/reference/provenancefiltering/NC2017_Dev1-provenancefiltering-ref-node.csv'
# journalRefPath = '/Users/joel/Documents/Projects/Medifor/results/reference/provenancefiltering/NC2017_Dev1-provenancefiltering-ref.csv'
#
# journalDir = '/Users/joel/Documents/Projects/Medifor/results/reference/provenancefiltering/journals'
# jsonDir = '/Users/joel/Documents/Projects/Medifor/results/json_old'
# runGraphSet(journalDir,jsonDir,worldIndexPath,nodeRefPath,journalRefPath)
parser = argparse.ArgumentParser()

parser.add_argument('--refFolder',help='folder that holds the reference files for provenance filtering, and the reference journals')
parser.add_argument('--jsonDir',help='folder that holds the jsons')
parser.add_argument('--worldIndex',help='world index csv file')

args = parser.parse_args()
jsonDir = args.jsonDir
worldIndexPath = args.worldIndex
refFolder = args.refFolder
journalDir = os.path.join(refFolder, 'provenancefiltering', 'journals')
refFileFolder = os.path.join(refFolder, 'provenancefiltering')
refFiles = os.listdir(refFileFolder)
nodeRefPath = None
journalRefPath = None
for f in refFiles:
    if f.endswith('node.csv'):
        nodeRefPath = os.path.join(refFileFolder, f)
    if f.endswith('ref.csv'):
        journalRefPath = os.path.join(refFileFolder, f)



runGraphSet(journalDir,jsonDir,worldIndexPath,nodeRefPath,journalRefPath)



