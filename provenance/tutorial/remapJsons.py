import os
import sys
import json

jsonDir = sys.argv[1]
worldDir = sys.argv[2]
worldImages = {}
probeImages = {}
for f in os.listdir(os.path.join(worldDir,'../probe')):
    file = os.path.basename(f)
    id = file.split('.')[0]
    probeImages[id] = file
for f in os.listdir(worldDir):
    file = os.path.basename(f)
    id = file.split('.')[0]
    worldImages[id] = file

for f in os.listdir(jsonDir):
    jsonFile = os.path.join(jsonDir,f)
    with open(jsonFile,'r') as fp:
        ranks = json.load(fp)
    nodes = ranks['nodes']
    for n in nodes:
        file = n['fileid']
        id = file.split('.')[0]
        if id in worldImages:
            newfile = os.path.join('world', worldImages[id])
        elif id in probeImages:
            newfile = os.path.join('probe', probeImages[id])
        n['fileid'] = id
        n['file'] = newfile
    with open(jsonFile,'w') as fp:
        json.dump(ranks,fp)