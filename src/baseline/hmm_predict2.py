import librosa

import sys
sys.path.append('../data')

import load_melodies
import load_cqt

import numpy as np
import os
import pickle
import io

from sklearn import hmm


models_dir = '../../models/hmm'
melodies_folderpath = '/home/jmj418/melody_type1'
cqt_folderpath = '/home/jmj418/cqt2'
datasplits_filepath = '../data/datasplits.txt'
output_dir = '/home/jmj418/hmm_predictions'

def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj
    
def pickleIt(pyName, outputName):
    output = open(outputName+'.pk1', 'wb')
    pickle.dump(pyName, output)
    output.close()


A,means,covmats,num_samples = pickleLoad(os.path.join(models_dir,'hmm2'))

nclasses = A.shape[0]

ignore = np.array([covmats[note]==None for note in range(nclasses)])

A = A[ignore==False,:][:,ignore==False]
covmats = np.vstack([cov.reshape(1,cov.shape[0],cov.shape[1]) for cov in covmats if cov != None])
means = np.vstack([mu.reshape(1,len(mu)) for mu in means if mu != None])

A = A/np.sum(A,axis=1).reshape(-1,1)

assert A.shape[0] == covmats.shape[0],"A and covmats dim not aligned"
assert A.shape[0] == means.shape[0],"A and means dim not aligned"

startprob = np.zeros(A.shape[0])
startprob[0]=1.0

model = hmm.GaussianHMM(A.shape[0],"full",startprob,A)
model.means_ = means
model.covars_ = covmats
model.startprob_ = startprob

melodies = load_melodies.load_melodies(melodies_folderpath,datasplits_filepath)
CQT = load_cqt.load_cqt(cqt_folderpath,datasplits_filepath)


def predictz(cqt_list,melody_list,set_name):
    os.system('mkdir -p ' + os.path.join(output_dir,set_name))
    
    allcqts = None
    allmelodies = None
    allz = None
    
    for i,song in enumerate(cqt_list):
        print "generating allcqts and allmelodies arrays",i,"of",len(cqt_list)
        melody = melody_list[i]
        cqt = CQT.load(song)
        n,p = cqt.shape
        assert melody.shape[0]==n,"melody and cqt have different lengths, %s" % song
        
        z = model.decode(cqt)[1]
        
        numreducedclasses = sum(ignore==False)
        numskips = np.array([sum(ignore[:i+1]==False) for i in range(nclasses)])
        adj = np.array([max(sum(numskips<=i),0) for i in range(numreducedclasses)])
        z_adj = np.array([adj[note] for note in z]) 
        
        if allmelodies == None:
            allmelodies = melody
            allcqts = cqt
            allz = z_adj
        else:
            allmelodies = np.hstack((allmelodies,melody))
            allcqts = np.vstack((allcqts,cqt))
            allz = np.hstack((allz,z_adj))
        
        np.savetxt(os.path.join(output_dir,set_name,song+'.csv'),z_adj,fmt='%d',delimiter=',')
        
    pickleIt(allmelodies,os.path.join(output_dir,set_name+"_melodies"))
    pickleIt(allz,os.path.join(output_dir,set_name+"_z"))
    
    voice_det_confusion = np.zeros((2,2))
    voice_det_confusion[0,0] = np.sum(np.logical_and(allz>0,allmelodies>0))
    voice_det_confusion[0,1] = np.sum(np.logical_and(allz==0,allmelodies>0))
    voice_det_confusion[1,0] = np.sum(np.logical_and(allz>0,allmelodies==0))
    voice_det_confusion[1,1] = np.sum(np.logical_and(allz==0,allmelodies==0))
    
    voice_det_accuracy = np.mean((allz>0) == (allmelodies>0))*100.0
    pitch_track_accuracy = np.mean((allz[allmelodies>0] == allmelodies[allmelodies>0]))*100.0
    overall_accuracy = np.mean((allz == allmelodies))*100.0
    
    return (voice_det_accuracy,pitch_track_accuracy,overall_accuracy),voice_det_confusion
        
train_accuracies,train_voice_det_conf = predictz(CQT.splits.train,melodies.train_list,'train')
validation_accuracies,validation_voice_det_conf = predictz(CQT.splits.validation,melodies.validation_list,'validation')

report = []
report.append("\ntrain: voicing %.1f%%, pitch %.1f%%, overall %.1f%%" % train_accuracies)
report.append('\nconfusion matrix (count):\n')
report.append(train_voice_det_conf)
report.append('\nconfusion matrix (%):\n')
report.append(np.round(train_voice_det_conf*100.0/np.sum(train_voice_det_conf),1))
report.append("\nvalidation: voicing %.1f%%, pitch %.1f%%, overall %.1f%%" % validation_accuracies)
report.append('\nconfusion matrix:\n')
report.append(validation_voice_det_conf)
report.append('\nconfusion matrix (%):\n')
report.append(np.round(validation_voice_det_conf*100.0/np.sum(validation_voice_det_conf),1))

with open(os.path.join(output_dir,'report.txt'),'w+') as file:
    for r in report:
        print r
        print >>file,r


pickleIt((train_accuracies,validation_accuracies),os.path.join(output_dir,'accuracies'))
    


def plotz(z,cqt,melody,window=range(2000,3000)):
    import matplotlib.pyplot as plt

    plt.pcolor(cqt[:,window],cmap='gnuplot2')
    plt.hlines(melody[window], np.arange(len(window)),np.arange(len(window))+1,
               color='r',linewidths=5)
    plt.hlines(z[window], np.arange(len(window)),np.arange(len(window))+1,
               color='w',linewidths=2)
    plt.xlim(0,len(window))
    plt.ylim(0,cqt.shape[0])
    plt.title("cqt spectrogram with melody (red)")
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    plt.show()



