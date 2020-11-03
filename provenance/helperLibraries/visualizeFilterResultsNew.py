import json
import sys
import os
import argparse
import urllib.request
filedict_rel = {}
def buildFilterPage(jsonFile,DatasetName,ServerPort,k,ServerAddress='http://medifor2.ecn.purdue.edu',numColumns = 5):
    head = "<head>\n\
    <style>\n\
    .container {\n\
      height: 20%;\n\
      border: dashed blue 1px;\n\
    \n\
    }\n\
    .root {\n\
      height: 20%;\n\
      border: dashed red 1px;\n\
    \n\
    }\n\
    \n\
    .container img {\n\
      max-height:100%;\n\
      max-width: 100%;\n\
    }\n\
    .root img {\n\
      max-height:100%;\n\
      max-width: 100%;\n\
    }\n\
    #parent {\n\
      display: flex;\n\
      flex-flow: row;\n\
      width: 100%;\n\
      height: auto; \n\
    }\n\
    </style>\n\
    </head>\n"


    with open(jsonFile,'r') as fp:
        results = json.load(fp)
    nodes=results['nodes']

    imageList = ""

    ncount = 1
    outputName = os.path.basename(jsonFile).split(".")[0]+".html"
    imageList += "<body>"
    rootid = os.path.basename(jsonFile)[:-5]

    if rootid in filedict_rel :
        rootNode = filedict_rel[rootid]
    else:
        rootNode = os.path.join('world',rootid)
    imageList += "<div id=\"parent\">\n"
    imageList += "<div class=\"root\">\n\
                        <img src=\"" + str(ServerAddress)+":"+str(ServerPort)+"/"+os.path.join(DatasetName,rootNode) + "\" />\n\
                      </div>"
    for node in nodes[:k]:
        if ncount % numColumns == 0 and ncount > 0:
            imageList += '</div>\n'
        if ncount%numColumns == 0 and ncount > 0 and ncount < len(nodes)-1:
            imageList += "<div id=\"parent\">\n"
        ncount+=1
        fname = node["file"]
        fid = os.path.basename(fname).split('.')[0]
        fpath = None
        if fid in filedict_rel:
            fname = filedict_rel[fid]
            fpath = fname
            #print('fpath: ',fpath)
        if not fpath: fpath = fname
        imageList += "<div class=\"container\">\n\
                <img src=\"http://{}:{}/".format(ServerAddress, ServerPort) + os.path.join(fpath) + "\" />\n\
                      </div>"
    imageList += "</body>"
    return head+imageList

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters")
    parser.add_argument("--datasetName")
    parser.add_argument("--recall", type=int)
    parser.add_argument("--address",type=str,default='0.0.0.0')
    parser.add_argument("--port", type=int,default=8001)
    parser.add_argument('--folder', action='store_true')
    parser.add_argument('--outputDir')
    parser.add_argument('--filedict',default=None)
    args = parser.parse_args()
    filedict = None
    if args.filedict is not None:
        with open(args.filedict,'r') as fp:
            filedicttmp = json.load(fp)
        #filedict_rel = {}
        dname = args.datasetName
        for f in filedicttmp:
            fullpath = filedicttmp[f]
            parts = fullpath.split('/')
            i = 0
            if dname in parts:
                i = min(parts.index(dname)+1,len(parts)-1)
            relpath = urllib.request.pathname2url('/'.join(parts[i:]))
            #relpath = '/'.join(parts[i:]).replace('(','%28').replace(')','%29')
            filedict_rel[f]= relpath
            filedict_rel[os.path.splitext(f)[0]] = relpath
        #for f in list(filedict_rel.keys())[:10]:
            #print(f)
            #print(filedict_rel.keys())
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    if args.folder is True:
        for jfile in os.listdir(args.jsonFile):
            if jfile.endswith('.json'):
                htmlTXT = buildFilterPage(os.path.join(args.jsonFile,jfile),args.datasetName,args.port,args.recall,args.address)
                with open(os.path.join(args.outputDir,jfile+'.html'),'w') as f:
                    f.write(htmlTXT)
    else:
        htmlTXT = buildFilterPage(args.jsonFile,args.datasetName,args.port,args.recall,args.address)
        with open(os.path.join(args.outputDir, os.path.basename(args.jsonFile) + '.html'), 'w') as f:
            f.write(htmlTXT)
