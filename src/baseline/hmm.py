import sys
sys.path.append('../data')

import datasplits
import data_proc
    

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

train,validation,test = datasplits.readsplits(path='../data/datasplits.txt')


melody_path = '../../../melody_type1'

melody_list_train = []
melody_list_validation = []

for t in train:
    melody_list_train.append(np.loadtxt(os.path.join(melody_path,t+'.csv'),delimiter=','))
for t in validation:
    melody_list_validation.append(np.loadtxt(os.path.join(melody_path,t+'.csv'),delimiter=','))
    
melodies = data_proc.melody_to_midi(np.hstack(melody_list_train),fmin=32.7023)
melodies_validation = data_proc.melody_to_midi(np.hstack(melody_list_validation),fmin=32.7023)