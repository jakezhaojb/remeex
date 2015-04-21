import sys
sys.path.append('../data')

import load_melodies
import load_cqt

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import gc
import os
import pickle

from scipy.stats import multivariate_normal


models_dir = '../../models/hmm'

def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj


A,covmats,means = pickleLoad(os.path.join(models_dir,'hmm'))

melodies_folderpath = '../../../melody_type1'
cqt_folderpath = '../../../cqt2'
datasplits_filepath = '../data/datasplits.txt'
plots_dir = '../../plots/'
models_dir = '../../models/hmm'

melodies = load_melodies.load_melodies(melodies_folderpath,datasplits_filepath)
CQT = load_cqt.load_cqt(cqt_folderpath,datasplits_filepath)

songnum = 1

melody = melodies.validation_list[songnum]
cqt = CQT.load(CQT.splits.validation[songnum])

window = range(0,2000)
plt.pcolor(cqt[window].T,cmap='gnuplot2')
plt.hlines(melody[window], np.arange(len(window)),np.arange(len(window))+1,
           color='r',linewidths=5)
plt.xlim(0,len(window))
plt.ylim(0,cqt.shape[1])
plt.title("cqt spectrogram with melody (red)")
fig = plt.gcf()
fig.set_size_inches(18.5,10.5)
plt.show()


nclasses = A.shape[0]

startprob = np.zeros(nclasses)
startprob[0]=1.0

X = np.zeros((cqt.shape[0],len(means[0])))
V = np.zeros((cqt.shape[0],nclasses))

for j in range(nclasses):
    if covmats[0] != None:
        print j,la.det(covmats[0])
    else:
        print j


def mn(x,mean,cov):
    n = len(mean)
    v = x - mean
    return (2*pi)**(-n/2.0)*np.exp(np.log(np.linalg.det(cov)**(-0.5))-0.5*np.dot(np.dot(v.T,np.linalg.inv(cov)),v))








