best version: hmm2.pk1   (A,means,covmats,num_samples)

older versions:

hmm.pk1    (A,means,covmats)

load using the following commands:

'
import os
import pickle


models_dir = '../../models/hmm'

def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj

A,means,covmats,num_samples = pickleLoad(os.path.join(models_dir,'hmm2'))
'