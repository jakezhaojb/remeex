import datasplits
import data_proc
import numpy as np
import os

class load_melodies():
    train = None
    validation = None
    test = None
    melody_folderpath = None
    datasplit_filepath = None
    
    def __init__(self,melody_folderpath,datasplit_filepath='datasplits.txt'):
        train,validation,test = datasplits.readsplits(path=datasplit_filepath)
        
        self.melody_folderpath = '../../../melody_type1'
        self.datasplit_filepath = datasplit_filepath
        
        melody_list_train = []
        melody_list_validation = []
        melody_list_test = []
        
        for t in train:
            melody_list_train.append(np.loadtxt(os.path.join(self.melody_folderpath,t+'.csv'),delimiter=','))
        for t in validation:
            melody_list_validation.append(np.loadtxt(os.path.join(self.melody_folderpath,t+'.csv'),delimiter=','))
        for t in test:
            melody_list_test.append(np.loadtxt(os.path.join(self.melody_folderpath,t+'.csv'),delimiter=','))
            
        self.train = data_proc.melody_to_midi(np.hstack(melody_list_train),fmin=32.7023)
        self.validation = data_proc.melody_to_midi(np.hstack(melody_list_validation),fmin=32.7023)
        self.test = data_proc.melody_to_midi(np.hstack(melody_list_validation),fmin=32.7023)