import os
import json
import argparse

from tqdm import tqdm

from resources import Resource
from fileutil import isRaw, make_sure_path_exists
from queryIndex import queryIndex

def convertNISTJSON(results):
    jsonResults={}
    nodes=[]
    jsonResults['directed']=True

    scores = results['provenance_filtering']['matches']
    meta = results['provenance_filtering']['meta']
    count=1
    for filename in scores:
        node={}
        node['id']= str(count)
        node['file']='world/'+filename
        node['fileid']= os.path.splitext(os.path.basename(filename))[0]
        node['nodeConfidenceScore']= scores[filename]
        #if meta is not None:
        #    node['meta'] = meta[filename]
        nodes.append(node)
        count=count+1
    jsonResults['nodes']=nodes
    jsonResults['links']=[]
    jsonstring = json.dumps(jsonResults)
    return jsonstring

def query_image(query_image, recall, rootpath='', index):
    probeFilename  = query_image.key

    #this can be called as many times as needed
    #image files will be put in
    allResults = []
    allResults.append(filteringResults())
    query_result = index.queryImage(query_image, recall, rootpath=rootpath)
    allResults[0].mergeScores(query_result)

    #Tier2
    maxScore = allResults[0].scores[list(allResults[0].scores.keys())[0]]

    if self.MultiTier and maxScore > .03: #only do multitier search if the first query gets enough votes (3% of all features match)
        try:
            mainResult = allResults[0]
            tier2ImageResources = []
            for r in list(mainResult.scores):
                tier2ImageResources.append(self.scalableQuery.getWorldImage(r))
            fullTier2FeatureResource,tier2FeatureSets, featureIDList, featureObjectIDList, featureDictionary,queryOrResultList,featureSetMap,visDict = multiTierFeatureBuilder.getTier2Features(probeImage,tier2ImageResources,30)
            if fullTier2FeatureResource is not None:
              # allTier2Results = self.scalableQuery.queryFeatures([fullTier2FeatureResource['supplemental_information']['value']], 100,ignoreIDs=list(allResults[0].map))
              allTier2Results = self.scalableQuery.queryFeatures(tier2FeatureSets,75,ignoreIDs=list(allResults[0].map))
              print('found results for ',len(allTier2Results),' tier 2 objects')
              # allTier2Scores = multiTierFeatureBuilder.getObjectScores(allTier2Results[0],featureIDList,featureObjectIDList,featureDictionary,queryOrResultList,objectWise=True,ignoreIDs=list(allResults[0].map))
              allTier2Scores = allTier2Results
              finalTier2Ranks = filteringResults()
              for r in allTier2Scores:
                  r.I = None
                  r.D = None
                  r.pairDownResults(2)
              print('merging tier 2 scores')
              for r in allTier2Scores:
                  finalTier2Ranks.mergeScores(r,ignoreIDs=allResults[0].map)
              # scoreMerge.mergeScoreSet(allTier2Scores)
              allResults[0].mergeScores(finalTier2Ranks)
            else:
                allTier2Results = None
        except:
            print('failed tier 2 search')
            allTier2Results = None
    # print(allResults)
    outputJson = self.createOutput(probeFilename,allResults[0])

    return outputJson

def query_images(image_list, recall, index):
    count = 0
    for query_path in tqdm(query_paths):
        doRun = True
        count +=1
        filepath = query_path.rstrip()
        filename = os.path.basename(filepath)

        fileID = os.path.splitext(filename)[0]
        resultDir = os.path.dirname(args.ProvenanceOutputFile)

        try:
            os.makedirs(os.path.dirname(args.ProvenanceOutputFile))
        except:
            pass

        jsonFile = 'json/' + fileID + '.json'
        jsonPath = os.path.join(resultDir, jsonFile)

        if not os.path.exists(jsonPath):
            query_image = Resource.from_file(filename, filepath)

            try:
                results = query_image(query_image, int(args.Recall), rootpath=os.path.dirname(filepath), index)

                if 'empty' in results['provenance_filtering']['matches']:
                    print(f'[WARNING]: Query {filepath} returned 0 results, skipping...')
                    continue

                # jsonString = convertNISTJSON(results)
                jsonString = results

                with open(jsonPath, 'w') as jf:
                    print(f'[LOG]: Writing json for query {filepath}...')
                    jf.write(jsonString)

            except Exception as e:
                print(f'[ERROR]: There was an error processing query file {filepath}...\n\t[DETAILS]: {e}')
        else:
            print(f'[WARNING]: Query result file already exists for {filepath}, skipping...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--QueryImages', help = 'List of images (paths) to query, one per line', default=None)
    parser.add_argument('--det', help = 'feature detector', default=None)
    parser.add_argument('--desc', help = 'feature descriptor', default=None)
    parser.add_argument('--outputdir', help = 'where to output query results', default=None)
    parser.add_argument('--IndexOutputDir', help='directory containing Indexes',default=None)
    parser.add_argument('--ProvenanceOutputFile', help='output directory for results')
    parser.add_argument('--Recall', help='output directory for the Index')
    parser.add_argument('--CacheFolder', help='location of cache folder',default='.')
    parser.add_argument('--TrainedIndexParams', help='location of index parameters file')
    args = parser.parse_args()

    make_sure_path_exists(os.path.join(os.path.dirname(args.ProvenanceOutputFile),'json'))

    query_paths = []
    with open(args.QueryImages) as f:
        query_paths = f.readlines()

    numcores = 1

    index = queryIndex(args.IndexOutputDir, indexParamFile=args.TrainedIndexParams,
                det=args.det, desc=args.desc, savefolder=savefolder)

    query_images(query_paths, args.Recall, index)
