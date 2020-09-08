import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--ImageList', help='list of all dataset images')
parser.add_argument('--OutputProbeList', help='location of output list of probe files')
parser.add_argument('--NumSamples', help='Number of random probes to produce')

args = parser.parse_args()

with open(args.ImageList,'r') as fp:
    lines = fp.readlines()

lines2 = []
for l in lines:
    lines2.append(l.rstrip())

lines2 = np.array(lines2)
ilist = np.random.choice(len(lines2),replace=False,size=int(args.NumSamples))
probelist = list(lines2[ilist])

with open(args.OutputProbeList,'w') as fp:
    fp.write('\n'.join(probelist))

