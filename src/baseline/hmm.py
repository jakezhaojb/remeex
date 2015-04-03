import sys
sys.path.append('../data')

import load_melodies

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

melodies_folderpath = '../../../melody_type1'
datasplits_filepath = '../data/datasplits.txt'

melodies = load_melodies.load_melodies(melodies_folderpath,datasplits_filepath)

nclasses = 85

A = np.zeros((nclasses,nclasses))

for t in melodies.train_list:
    tl = t[:-1]
    tr = t[1:]
    
    