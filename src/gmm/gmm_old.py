import librosa
import sys
sys.path.append('../data')
import load_melodies
import load_cqt
sys.path.append('../')
import utils
import numpy as np
import gc
import os
import pickle
from sklearn import mixture

cqt_folderpath = '/home/jmj418/cqt2'
datasplits_filepath = '../data/datasplits.txt'
plots_dir = '../../plots/'
models_dir = '../../models/gmm'

nComponents = 64
kX = 5; dX = 1
kY = 5; dY = 1

print '\n>> nComponents=%d kX=%dx%d kY=%dx%d' % (nComponents,kX,dX,kY,dY),'\n'

CQT = load_cqt.load_cqt(cqt_folderpath,datasplits_filepath)
inputs = []

for i,song in enumerate(CQT.splits.train):
    print ">> generating cqt arrays",i,"of",len(CQT.splits.train)
    cqt = CQT.load(song)
    n,p = cqt.shape

    iX = np.arange(0,n-kX+1,dX)
    iY = np.arange(0,p-kY+1,dY)

    for x in iX:
        for y in iY:
            inputs.append(cqt[x:x+kX,y:y+kY].ravel())
print '\n>>',len(inputs),"inputs"

np.random.seed(1)
g = mixture.GMM(n_components=nComponents)

inputs = np.vstack(inputs)

print "fitting gmm"
g.fit(inputs)

filename = 'gmm_n%d_%dx%d_%dx%d' % (nComponents,kX,dX,kY,dY)
path = os.path.join(models_dir,filename)

utils.pickleIt(g,path)




