import sys
sys.path.append('../data')

import datasplits
import data_proc

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

train,validation,test = datasplits.readsplits(path='../data/datasplits.txt')

melody_path = '../../../melody_type1'

melody_list_train = []
melody_list_validation = []

for t in train:
    melody_list_train.append(np.loadtxt(os.path.join(melody_path,t+'.csv'),delimiter=','))
for t in validation:
    melody_list_validation.append(np.loadtxt(os.path.join(melody_path,t+'.csv'),delimiter=','))
    
melodies = data_proc.melody_to_midi(np.hstack(melody_list_train),fmin=32.7023)
melodies_validation = data_proc.melody_to_midi(np.hstack(melody_list_validation),fmin=32.7023)

plt.hist(melodies)
plt.title('histogram of all melody labels\n(train split only)')
plt.savefig('../../plots/histogram_melody_all.jpg')
plt.show()

n,bins,_ = plt.hist(melodies[melodies>0],bins=range(85))
majority_class = np.arange(85)[n==max(n)][0]
majority_class_prob_train = n[majority_class]*1.0/sum(n)*100
majority_class_prob_valid = sum(melodies_validation==majority_class)*1.0/sum(melodies_validation>0)
title = 'histogram of melody present\n'
title = title + 'majority class=%d\n' % majority_class
title = title + 'train: P(y=38|melody present)=%.2f%%\n' % majority_class_prob_train
title = title + 'validation: P(y=38|melody present)=%.2f%%' % majority_class_prob_valid
plt.title(title)
plt.savefig('../../plots/histogram_given_melody.jpg')
plt.show()

no_melody = np.sum(melodies == 0)*1.0/len(melodies)
contains_melody = 1-no_melody

no_melody_validation = np.sum(melodies_validation == 0)*1.0/len(melodies_validation)
contains_melody_validation = 1-no_melody_validation

labels = ['no melody','has melody']
plt.bar(np.arange(2),[no_melody,contains_melody],width=0.4,label='train')
plt.bar(np.arange(2)+0.4,[no_melody_validation,contains_melody_validation],color=sns.xkcd_rgb["pale red"],width=0.4,label='validation')
plt.xticks([0.4,1.4],labels)
plt.legend()
plt.title('proportion of points with and without melody\n(train split only)')
plt.savefig('../../plots/melody_vs_no_melody.jpg')
plt.show()