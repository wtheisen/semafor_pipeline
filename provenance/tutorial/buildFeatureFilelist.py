import os
import sys
import progressbar
outerDirectory = sys.argv[1]
outputFile = sys.argv[2]
folders = []
folders2 = []
fileList = []
for folder in os.listdir(outerDirectory):
    fpath = os.path.join(outerDirectory,folder)
    if os.path.isdir(fpath):
        folders.append(fpath)
for folder in folders:
    for folder2 in os.listdir(folder):
        fpath = os.path.join(folder,folder2)
        if os.path.isdir(fpath):
            folders2.append(fpath)
for folder in folders2:
    files = os.listdir(folder)
    bar = progressbar.ProgressBar()
    for f in bar(files):
        if f.endswith('.npy'):
            fullFile = os.path.join(folder,f)
            fileList.append(fullFile)
with open(outputFile,'w') as fp:
    fp.write('\n'.join(fileList))