import os
import sys

baseFolder = sys.argv[1]
outputFile = sys.argv[2]

outsideFolders = os.listdir(baseFolder)
allFiles = []
for ofolder1 in outsideFolders:
    ofoPath = os.path.join(baseFolder,ofolder1)
    if os.path.isdir(ofoPath):
        for ofolder2 in os.listdir(ofoPath):
            ofoPath2 = os.path.join(ofoPath,ofolder2)
            if os.path.isdir(ofoPath2):
                files =os.listdir(ofoPath2)
                for f in files:
                    fname = os.path.join(ofoPath2,f)
                    if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg') or fname.lower().endswith('.png') or fname.lower().endswith('.gif'):
                        allFiles.append(fname)
with open(outputFile,'w') as fp:
    fp.write('\n'.join(allFiles))
