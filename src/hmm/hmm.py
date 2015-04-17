import librosa

import sys
sys.path.append('../data')
sys.path.append('../')

import load_melodies
import load_cqt

import numpy as np
import gc
import os
import utils
  

def generate_hmm_params(binary=False,
						filename='hmm.pk1',
						melodies_folderpath = '/home/jmj418/melody_type1',
						cqt_folderpath = '/home/jmj418/cqt2',
						datasplits_filepath = '../data/datasplits.txt',
						models_dir = '../../models/hmm'):

	nclasses = 85
	nCQTbins = 84 

	if binary:
		nclasses = 2

	melodies = load_melodies.load_melodies(melodies_folderpath,datasplits_filepath)

	IJ = 0
	for i,t in enumerate(melodies.train_list):
	    print "melody",i+1,"of",len(melodies.train_list)
	    if binary:
	    	t = (t>0).astype('int')
	    I = t[:-1].reshape(-1,1,1)==np.arange(nclasses).reshape(1,-1,1)
	    J = t[1:].reshape(-1,1,1)==np.arange(nclasses).reshape(1,1,-1)
	    IJ = IJ + np.sum(np.logical_and(I,J),axis=0)

	I_sum = np.sum(IJ,axis=1).reshape(-1,1).astype('float')

	I_sum[I_sum==0] = 0.1

	A = IJ*1.0/(I_sum)

	CQT = load_cqt.load_cqt(cqt_folderpath,datasplits_filepath)

	allcqts = None
	allmelodies = None

	for i,song in enumerate(CQT.splits.train):
	    print "generating allcqts and allmelodies arrays:",i,"of",len(CQT.splits.train)
	    melody = melodies.train_list[i]
	    if binary:
	    	melody = (melody>0).astype('int')
	    cqt = CQT.load(song)
	    n,p = cqt.shape
	    assert melody.shape[0]==n,"melody and cqt have different lengths, %s" % song
	    
	    if allmelodies == None:
	        allmelodies = melody
	        allcqts = cqt
	    else:
	        allmelodies = np.hstack((allmelodies,melody))
	        allcqts = np.vstack((allcqts,cqt))
	    
	        
	means = []
	covmats = []
	num_samples = []


	for note in range(nclasses):
	    gc.collect()
	    print "generating means and covmats",note,"of",nclasses
	    sel = allmelodies == note
	    n_sel = sum(sel)
	    num_samples.append(n_sel)
	    
	    if n_sel > 1:
	        means.append(np.mean(allcqts[sel],axis=0))
	        covmats.append(np.cov(allcqts.T))
	    else:
	        means.append(None)
	        covmats.append(None)
	        

	hmm_params = (A,means,covmats,num_samples)
	utils.pickleIt(hmm_params,os.path.join(models_dir,filename))

