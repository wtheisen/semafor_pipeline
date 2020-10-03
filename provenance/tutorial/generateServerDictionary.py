import os
import sys
import json
import progressbar

def generateFileDictionary(dirList):
    nameToPathDictionary = {}
    nameToIDDictionary = {}
    IDToNameDictionary = {}
    fileList = []
    count = 0
    try:
        for directoryToIndex in dirList:
            print("searching " + directoryToIndex)
            notfindFileTypes = True
            bar = progressbar.ProgressBar()
            for root, dirs, files in bar(os.walk(directoryToIndex,followlinks=True)):
                dlength = len(dirs)
                for file in files:
                    fileID = file.split('.')[0]
                    if notfindFileTypes or file.endswith('.npy'):
                        print(fileID)

                        if fileID not in nameToPathDictionary:
                            f = os.path.join(os.path.abspath(root),file)
                            nameToPathDictionary[fileID] = f
                            fileList.append(f)
                            count += 1
                        else:
                            pass
                            # print('file already exists, skip')
                        #if count%1000 == 0:
                        #    print(str(count)+"/"+str(dlength))
            nameToPathDictionary["_rootDirectory_"] = os.path.abspath(directoryToIndex)
    except Exception as e:
        print(e)

    return nameToPathDictionary,fileList

try:
    dirPathList = sys.argv[1].split(',')
    outputName = sys.argv[2]
except Exception as e:
    print(e)

serverDict,fileList = generateFileDictionary(dirPathList)

outputPath = os.path.join(outputName+"_pathmap.json")
outputListPath = os.path.join(outputName+"_filelist.txt")
print(outputPath)
with open(outputPath,'w') as f:
    json.dump(serverDict,f)
with open(outputListPath,'w') as fp:
    fp.write('\n'.join(fileList))
