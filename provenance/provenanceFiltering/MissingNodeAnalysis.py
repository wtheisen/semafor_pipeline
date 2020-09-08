import os
import sys
import json
import numpy as np
import progressbar
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
def getMissingNodesFromExample(resultJson,gtJournal,world_index_list):
    foundNodes = resultJson['nodes']
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
    return missingNodeIndexes,correctNodeIndexes,erroneousNodeNames

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

def runGraphAnalysis(journalPath,jsonPath,worldIndexList,worldIndexExtMap,nodeMap,nodeMap_reverse):
    with open(journalPath, 'r') as fp:
        gtJournal = json.load(fp)
    with open(jsonPath, 'r') as fp:
        resultJson = json.load(fp)
    remapNodes(gtJournal, nodeMap)
    missingNodeIndexes, correctNodeIndexes, erroneousNodeIndexes = getMissingNodesFromExample(resultJson, gtJournal,
                                                                                              worldIndexList)
    graphItem = buildAdjListFromJson(gtJournal)
    probeName = nodeMap_reverse[worldIndexExtMap[os.path.basename(jsonPath).split('.')[0]]]

    headerLine = 'probeName,nodeName,isMissing,degree_in,degree_out,isLeaf,isRoot,isInProbeChain,isAboveProbe,isBelowProbe,isDonorToProbe,isReceiverOfProbe,isDonorAtSomePoint,distanceFromLeaf,distanceFromRoot,dfr_normalized,dfl_normalized,distanceFromProbe_undirected,distanceFromProbe_withChainJumps,chainJumps,minimumPenalty'
    csvOutput = []
    for i in missingNodeIndexes:
        fname = gtJournal['nodes'][i]['file']
        if fname in nodeMap_reverse:
            fname_r = nodeMap_reverse[fname]
            prefix_data = [os.path.basename(jsonPath),fname,True]
            allData = analyzeNodeInGraph(fname_r, probeName, graphItem)
            prefix_data.extend(allData)
            csvOutput.append(','.join(str(elem) for elem in prefix_data))
    for i in correctNodeIndexes:
        fname = gtJournal['nodes'][i]['file']
        if fname in nodeMap_reverse:
            fname_r = nodeMap_reverse[fname]
            prefix_data = [os.path.basename(jsonPath), fname, False]
            allData = analyzeNodeInGraph(fname_r, probeName, graphItem)
            prefix_data.extend(allData)
            csvOutput.append(','.join(str(elem) for elem in prefix_data))
    return csvOutput


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
    headerLine = 'probeName,nodeName,isMissing,degree_in,degree_out,isLeaf,isRoot,isInProbeChain,isAboveProbe,isBelowProbe,isDonorToProbe,isReceiverOfProbe,isDonorAtSomePoint,distanceFromLeaf,distanceFromRoot,dfr_normalized,dfl_normalized,distanceFromProbe_undirected,distanceFromProbe_withChainJumps,chainJumps,minimumPenalty'
    totalCSV = [headerLine]
    bar = progressbar.ProgressBar()
    for jsonGraphFile in bar(os.listdir(jsonDir)):
        jsonFile = os.path.join(jsonDir,jsonGraphFile)
        probeName = jsonGraphFile.split('.')[0]
        if probeName in probeToJournalMap:
            journalName = probeToJournalMap[probeName]
            journalPath = os.path.join(journalDir,journalName)
            allData = runGraphAnalysis(journalPath,jsonFile,worldIndexList,worldIndexExtMap,nodeMap,nodeMap_reverse)
            totalCSV.extend(allData)
    print('writing ', len(totalCSV),' lines...')
    with open('allOutput.csv','w') as fp:
        fp.write('\n'.join(totalCSV))


journalPath = '/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/reference/provenance/journals/88f18d781fdebd6836ac44fcaa656c1b.json'
jsonPath = '/home/jbrogan4/Desktop/NC2017_results/e68047950999e492e8a024958e54a83a.json'
worldIndexPath = '/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/indexes/NC2017_Dev1-provenancefiltering-world.csv'
nodeRefPath = '/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/reference/provenance/NC2017_Dev1-provenance-ref-node.csv'
journalRefPath = '/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/reference/provenance/NC2017_Dev1-provenance-ref.csv'

journalDir = '/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/reference/provenance/journals/'
jsonDir = '/home/jbrogan4/Desktop/NC2017_results/'

runGraphSet(journalDir,jsonDir,worldIndexPath,nodeRefPath,journalRefPath)

# with open(journalPath,'r') as fp:
#     gtJournal = json.load(fp)
# with open(jsonPath,'r') as fp:
#     resultJson = json.load(fp)
#
# with open(worldIndexPath,'r') as fp:
#     lines = fp.readlines()
# for l in lines[1:]:
#     parts = l.split('|')
#     if len(parts) > 1:
#         fname = os.path.basename(parts[2])
#         fname_noext = fname.split('.')[0]
#         worldIndexList.append(fname)
#         worldIndexExtMap[fname_noext] = fname
#
# with open(nodeRefPath,'r') as fp:
#     lines = fp.readlines()
# for l in lines:
#     l = l.rstrip()
#     parts = l.split('|')
#     if len(parts) > 1:
#         nodeMap[parts[3]] = os.path.basename(parts[2])
#         nodeMap_reverse[os.path.basename(parts[2])] = parts[3]
#
# remapNodes(gtJournal,nodeMap)
# missingNodeIndexes,correctNodeIndexes, erroneousNodeIndexes = getMissingNodesFromExample(resultJson,gtJournal,worldIndexList)
# graphItem = buildAdjListFromJson(gtJournal)
# probeName = nodeMap_reverse[worldIndexExtMap[os.path.basename(jsonPath).split('.')[0]]]
# headerLine = 'degree_in,degree_out,isLeaf,isRoot,isInProbeChain,isAboveProbe,isBelowProbe,isDonorToProbe,isReceiverOfProbe,isDonorAtSomePoint,distanceFromLeaf,distanceFromRoot,dfr_normalized,dfl_normalized,distanceFromProbe_undirected,distanceFromProbe_withChainJumps,chainJumps,minimumPenalty'
# csvOutput = headerLine
# print('missing nodes')
# for i in missingNodeIndexes:
#     fname = gtJournal['nodes'][i]['file']
#     if fname in nodeMap_reverse:
#         fname = nodeMap_reverse[fname]
#         allData = analyzeNodeInGraph(fname,probeName,graphItem)
#         csvOutput += '\n'+','.join(allData)
#     print((fname,i))
# # useIndex = 6
# # visited = dfs(graphItem.alist_down,6)
# # startFrom = gtJournal['nodes'][6]['file']
# # if startFrom in nodeMap_reverse:
# #     startFrom = nodeMap_reverse[startFrom]
# # print('starting from ',startFrom)
# #
#
# # getNodeID('5df42c12fa2790578eee96b6ec93fd01.jpg')
# # fullchain = list(dfs_toLeafs(graphItem.alist_up,9))
# # for c in fullchain:
# #     pathString = ''
# #     for v in c:
# #         fileName = gtJournal['nodes'][v]['file']
# #         if fileName in nodeMap_reverse:
# #             mappedFileName = nodeMap_reverse[fileName]
# #         else:
# #             mappedFileName = fileName
# #         pathString+=mappedFileName+' -> '
# #     print(pathString)


