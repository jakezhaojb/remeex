import numpy as np
import os
import pickle

from sklearn import hmm


models_dir = '../../models/hmm'

def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj


A,covmats,means,num_samples = pickleLoad(os.path.join(models_dir,'hmm'))

nclasses = A.shape[0]

ignore = np.array([covmats[note]==None for note in range(nclasses)])

A = A[ignore==False,:][:,ignore==False]
covmats = np.vstack([cov.reshape(1,cov.shape[0],cov.shape[1]) for cov in covmats if cov != None])
means = np.vstack([mu.reshape(1,len(mu)) for mu in means if mu != None])


assert A.shape[0] == covmats.shape[0],"A and covmats dim not aligned"
assert A.shape[0] == means.shape[0],"A and means dim not aligned"

startprob = np.zeros(A.shape[0])
startprob[0]=1.0

model = hmm.GaussianHMM(A.shape[0],"full",startprob,A)
model.means_ = means
model.covars_ = covmats
model.startprob_ = startprob





