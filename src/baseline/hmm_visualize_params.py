import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
import pickle


models_dir = '../../models/hmm'
plots_dir = '../../plots/'

def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj


A,means,covmats,num_samples = pickleLoad(os.path.join(models_dir,'hmm'))

nclasses = A.shape[0]

plt.pcolor(A,cmap='gnuplot2')
plt.xlim(0,nclasses)
plt.ylim(0,nclasses)
plt.title('hmm transition matrix')
plt.gca().invert_yaxis()
plt.savefig(os.path.join(plots_dir,'hmm_transition_matrix.jpg'))
plt.show()

plt.bar(np.arange(nclasses),A[30])
plt.title('HMM transition matrix: P( j | i=30 )')
plt.savefig(os.path.join(plots_dir,'hmm_row30.jpg'))
plt.show()

plt.bar(np.arange(nclasses),A[0])
plt.title('HMM transition matrix: P( j | i=0 )')
plt.savefig(os.path.join(plots_dir,'hmm_row0.jpg'))
plt.show()


def plotnote(note):
    plt.bar(np.arange(len(means[note])),means[note])
    plt.xlim(0,len(means[note]))
    plt.title('hmm: cqt means,note=%d\n(train)' % note)
    plt.savefig(os.path.join(plots_dir,'hmm_cqt_means%d.jpg' % note))
    plt.show()
    
    plt.pcolor(covmats[note],cmap='gnuplot')
    plt.xlim(0,len(means[note]))
    plt.ylim(0,len(means[note]))
    plt.title('hmm: cqt covariance matrix,note=%d\n(train)' % note)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(plots_dir,'hmm_cqt_cov%d.jpg' % note))
    plt.show()
    
plotnote(0)
plotnote(30)



