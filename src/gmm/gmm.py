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
from sklearn import mixture

class args():
    def __init__(self,kX,dX,kY,dY,split):
        self.kX=kX
        self.dX=dX
        self.kY=kY
        self.dY=dY
        self.split=split


datadir = '/scratch/jmj418/data/gmm/'
models_dir = '../../models/gmm'

nComponents = 16
kX = 5; dX = 1
kY = 5; dY = 1

filename = 'obs_%dx%d_%dx%d' % (kX,dX,kY,dY)

print '\n>> nComponents=%d kX=%dx%d kY=%dx%d' % (nComponents,kX,dX,kY,dY),'\n'

np.random.seed(1)
g = mixture.GMM(n_components=nComponents)

print '>> load data'
path = os.path.join(datadir,filename)
shape,dtype,opt = utils.pickleLoad(path+'.pk1')
data = np.fromfile(os.path.join(datadir,filename)+'.npy',dtype=dtype).reshape(shape)
print '>>',data.shape

print ">> fitting gmm"
g.fit(data)

filename = 'gmm_n%d_%dx%d_%dx%d' % (nComponents,kX,dX,kY,dY)
path = os.path.join(models_dir,filename)

utils.pickleIt(g,path)




