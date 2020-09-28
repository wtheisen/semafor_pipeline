from provenanceFiltering import  provenanceFiltering,filteringResults
from distributedQuery import distributedQuery
import argparse
import os
from resources import Resource
import json
from fileutil import isRaw,make_sure_path_exists

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


parser = argparse.ArgumentParser()
parser.add_argument('--NISTProbeFileList', help='provenance index file', default=None)
parser.add_argument('--GenericProbeList', help = 'Probe List for non-NIST applications', default=None)
parser.add_argument('--det', help = 'feature detector', default=None)
parser.add_argument('--desc', help = 'feature descriptor', default=None)
parser.add_argument('--outputdir', help = 'where to output query results', default=None)
parser.add_argument('--NISTDataset', help='nist dataset directory')
parser.add_argument('--IndexOutputDir', help='directory containing Indexes',default=None)
parser.add_argument('--ProvenanceOutputFile', help='output directory for results')
parser.add_argument('--Recall', help='output directory for the Index')
parser.add_argument('--FileDict', help='file dictionary mapping image IDs to their absolute path locations',default=None)
parser.add_argument('--CacheFolder', help='location of cache folder',default='.')
parser.add_argument('--TrainedIndexParams', help='location of index parameters file')
parser.add_argument('--Offset', help = 'start image offset number from begining of file list',default=0)
parser.add_argument('--IndexServerAddress',help='address of the server that performs feature queries',default=None)
parser.add_argument('--IndexServerPort', help='port of server that performs the feature queries',default='9999')
args = parser.parse_args()
if args.IndexServerAddress is None and args.IndexOutputDir is None:
    print('Error, you must specify either an index directory via --IndexOutputDir or an index server address via --IndexServerAddress')
    exit(1)
make_sure_path_exists(os.path.join(os.path.dirname(args.ProvenanceOutputFile),'json'))
useServer = False
if args.IndexServerAddress is not None:
    useServer = True
provenanceresults=open(args.ProvenanceOutputFile,'w')
provenanceresults.write('ProvenanceProbeFileID|ConfidenceScore|ProvenanceOutputFileName|ProvenanceProbeStatus\n')
provenanceresults.close()

filedict = None
if args.FileDict is not None:
    print('loading file dictionary...')
    with open(args.FileDict,'r') as fp:
        filedict=json.load(fp)

if args.NISTProbeFileList is not None:
    fileIndex = -1
    with open(args.NISTProbeFileList) as f:
      lines = f.readlines()
    values = lines[0].split('|')
    if fileIndex == -1:
        try:
            fileIndex = values.index('ProvenanceProbeFileName')
        except:
            if filedict is None:
                print('using WorldFileName')
                fileIndex = values.index('WorldFileName')
                print('column index: ', fileIndex)
            else:
                print('using WorldFileID')
                fileIndex = values.index('WorldFileID')
                print('column index: ', fileIndex)
    lines = lines[1:]
elif args.GenericProbeList is not None:
    with open(args.GenericProbeList) as f:
      lines = f.readlines()
numcores = 1
print(args.outputdir)
print("beginning dist query")
distQuery = distributedQuery(args.IndexOutputDir, os.path.join(args.NISTDataset, "world"),indexServerAddress=args.IndexServerAddress,indexServerPort=args.IndexServerPort,tmpDir=args.CacheFolder,indexParamFile=args.TrainedIndexParams,useServer=useServer,det=args.det,desc=args.desc,savefolder=args.outputdir)
print("after dist query")
provenanceFilter = provenanceFiltering(distQuery)
print('after provenancefilter')
#provenanceFilter = None
count = 0
from tqdm import tqdm
for line in tqdm(lines[int(args.Offset):]):
    print("line", line)
    if count == 0:
        filepath = '/media/jbrogan4/bill/Pictures/InstagramImages/jokowilagi/51045154_603997083357112_1048945291012894074_n.jpg'

    doRun = True
    #print(count)
    count +=1

    if args.GenericProbeList is not None and args.NISTProbeFileList is None:
        #filename = os.path.basename(line)
        #fileid = os.path.splitext(filename)
        filepath = line.rstrip()
        if filedict is not None:
            filepath = filedict[values[fileIndex]]
    elif args.GenericProbeList is None and args.NISTProbeFileList is not None:
         if fileIndex==-1:
             fileIndex=values.index('ProvenanceProbeFileName')
         else:
            if filedict is None:
                filepath = os.path.join(args.NISTDataset,os.path.basename(values[fileIndex]))
                if not os.path.exists(filepath):
                    print('trying png extension instead')
                    filename = os.path.basename(values[fileIndex]).split('.')[0] + '.png'
                    filepath = os.path.join(args.NISTDataset, filename)
                    if os.path.exists(filepath):
                        print('success in finding as png!')
                        doRun = True
            else:
                filepath = filedict[values[fileIndex]]

    if True:
        filename = os.path.basename(filepath)
        #print(filename)
        # 489e2636eebfa924d7e03b717b2cdd0e.jpg leaf feat (489e2636eebfa924d7e03b717b2cdd0e.json)
        # 011f279514af6abff335ed3e2b02ea2c.jpg (011f279514af6abff335ed3e2b02ea2c.json.html) runner with photoshoped fez man on shirt
        # 8b3c9021c7e6dda308cfe7c594dc79e4.jpg beach with trash
        # '030d521fca99981c442e60e4b2155699.jpg':  # '1f012075b4bab96e28f7d9fa9aa85a4f.jpg':
        if not filename == '56501284787768799019d68baf0fb743.jpg': #not filename == '8b3c9021c7e6dda308cfe7c594dc79e4.jpg':
            pass
        fileID = os.path.splitext(filename)[0]
        resultDir = os.path.dirname(args.ProvenanceOutputFile)
        try:
            os.makedirs(os.path.dirname(args.ProvenanceOutputFile))
        except:
            pass
        jsonFile = 'json/' + fileID + '.json'
        jsonPath = os.path.join(resultDir, jsonFile)
        if (not os.path.exists(jsonPath) and not isRaw(filename)) and doRun:
            #print('running the bit that writes the json...')
            #print(filename)

            probeImage = Resource.from_file(filename, filepath)

            #print('Before actual query')

            try:
                results = provenanceFilter.processImage(probeImage,int(args.Recall),rootpath=os.path.dirname(filepath))

                print('After actual query')


                if 'empty' in results['provenance_filtering']['matches']:
                    print('skipping because empty returned')
                    continue

                print('Writing provenance results or something')

                provenanceresults = open(args.ProvenanceOutputFile, 'a')
                provenanceresults.write(fileID+'|1.0|'+jsonFile+'|Processed\n')
                provenanceresults.close()

                jsonString = convertNISTJSON(results)
                print('*-*-* ')

                with open(jsonPath, 'w') as jf:
                    print('writing json...')
                    jf.write(jsonString)

                # except IOError as e:
                #    print('skipping')
                #    logging.info('skipping '+filepath);
                #    provenanceresults.write(
                #             fileID = os.path.splitext(os.path.basename(filename))[0]os.path.splitext(os.path.basename(values[fileIndex]))[0]+'|1.0||NonProcessed\n')
                # except Exception as e:
                #    print('skipping2')
                #    logging.error(traceback.format_exc())
                #    provenanceresults.write(os.path.splitext(os.path.basename(values[fileIndex]))[0]+'|1.0||NonProcessed\n')
            except:
                print('error processing query file')
        else:
            print('skipping...')
    #except Exception as e:
    #    print('ERROR: ', e)

