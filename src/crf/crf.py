import librosa

import sys
sys.path.append('../data')
sys.path.append('../')

import load_melodies
import load_cqt

import numpy as np
#import matplotlib.pyplot as plt

#from sklearn.svm import LinearSVC
#from sklearn.metrics import confusion_matrix
import os
import pystruct
from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM

run_name = 'hmm3_voicing_detection'

melodies_folderpath = '/home/jmj418/melody_type1'
cqt_folderpath = '/home/jmj418/cqt2'
datasplits_filepath = '../data/datasplits.txt'
models_dir = '../../models/hmm'
output_dir = os.path.join('/home/jmj418/hmm_predictions',run_name)

binary = False
maxsongs = 200

melodies = load_melodies.load_melodies(melodies_folderpath,datasplits_filepath)

CQT = load_cqt.load_cqt(cqt_folderpath,datasplits_filepath)



def loaddata(binary,CQT,CQT_LIST,MELODY_LIST,maxsongs):
	y = []
	X = []
	melody_set = set()
	for i,song in enumerate(CQT_LIST):
		if i<maxsongs:
			print "loading cqt and melody arrays:",i,"of",len(CQT_LIST)
			melody = MELODY_LIST[i]
			if binary:
				melody = (melody>0).astype('int')
			else:
				melody_set = melody_set.union(melody.tolist()) 
			cqt = CQT.load(song)
			n,p = cqt.shape
			assert melody.shape[0]==n,"melody and cqt have different lengths, %s" % song

			y.append(melody)
			X.append(cqt)
		else:
			break

	return np.array(X),np.array(y),melody_set

print '\n>> loading data'
X_train,y_train,melody_set = loaddata(binary,CQT,CQT.splits.train,melodies.train_list,maxsongs)
X_validation,y_validation,_ = loaddata(binary,CQT,CQT.splits.validation,melodies.validation_list,maxsongs)

if not binary:
	missing = [i for i in range(85) if i not in melody_set]
	if len(missing) > 0:
		placeholderX = []
		placeholderY = []
		for note in missing:
			placeholderX.append(np.zeros((1,X_train[0].shape[1])))
			placeholderY.append(np.array([note]))
		X_train = np.array(X_train.tolist()+placeholderX)
		y_train = np.array(y_train.tolist()+placeholderY)


print '\n>> training CRF'
# Train linear chain CRF
model = ChainCRF()
ssvm = OneSlackSSVM(model=model, C=.1, inference_cache=50, tol=0.1)
ssvm.fit(X_train, y_train)

print '\n>> predicting on validation set:'
print("Test score with chain CRF: %f" % ssvm.score(X_validation, y_validation))
