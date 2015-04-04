import librosa

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

melodies_folderpath = '../../../melody_type1'
cqt_folderpath = '../../../cqt2'
datasplits_filepath = '../data/datasplits.txt'
plots_dir = '../../plots/'
models_dir = '../../models/hmm'

nclasses = 85
nCQTbins = 84   
   

melodies = load_melodies.load_melodies(melodies_folderpath,datasplits_filepath)

IJ = 0
for t in melodies.train_list:
    I = t[:-1].reshape(-1,1,1)==np.arange(nclasses).reshape(1,-1,1)
    J = t[1:].reshape(-1,1,1)==np.arange(nclasses).reshape(1,1,-1)
    IJ = IJ + np.sum(np.logical_and(I,J),axis=0)

I_sum = np.sum(IJ,axis=1).reshape(-1,1).astype('float')

I_sum[I_sum==0] = 0.1

A = IJ*1.0/(I_sum)



CQT = load_cqt.load_cqt(cqt_folderpath,datasplits_filepath)

sum_outer_products = [0]*nclasses
sum_for_mean = [0]*nclasses
num_samples = [0]*nclasses
for i,song in enumerate(CQT.splits.train):
    gc.collect()
    melody = melodies.train_list[i]
    cqt = CQT.load(song)
    n,p = cqt.shape
    for note in range(nclasses):
        sel = melody==note
        n_sel = sum(sel)
        num_samples[note] += n_sel
        if n_sel > 0:
            means = np.mean(cqt[sel],axis=0)
            normalized = cqt[sel] - means
            outer_products = np.sum(normalized.reshape(n_sel,p,1)*normalized.reshape(n_sel,1,p),axis=0)            
            sum_outer_products[note] = sum_outer_products[note] + outer_products
            sum_for_mean[note] = sum_for_mean[note] + means

covmat = []
means = []
for i in range(nclasses):
    if num_samples[i] > 1:
        covmat.append(sum_outer_products[i]/(num_samples[i]-1))
    else:
        covmat.append(None)
    if num_samples[i] > 0:
        means.append(sum_for_mean[i]/num_samples[i])
    else:
        means.append(None)

def plotnote(note):
    plt.bar(np.arange(len(means[note])),means[note])
    plt.xlim(0,len(means[note]))
    plt.title('hmm: cqt means,note=%d\n(train)' % note)
    plt.savefig(os.path.join(plots_dir,'hmm_cqt_means%d.jpg' % note))
    plt.show()
    
    plt.pcolor(covmat[note],cmap='gnuplot')
    plt.xlim(0,len(means[note]))
    plt.ylim(0,len(means[note]))
    plt.title('hmm: cqt covariance matrix,note=%d\n(train)' % note)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(plots_dir,'hmm_cqt_cov%d.jpg' % note))
    plt.show()
    
plotnote(0)
plotnote(30)



def pickleIt(pyName, outputName):
    output = open(outputName+'.pk1', 'wb')
    pickle.dump(pyName, output)
    output.close()

hmm_params = (A,means,covmat)
pickleIt(hmm_params,os.path.join(models_dir,'hmm2'))


note = 0
sel = melody == note
std = np.std(cqt[sel]/1000,axis=0)
plt.bar(np.arange(len(std)),std)

y = np.diag(std[20:60]**2)



