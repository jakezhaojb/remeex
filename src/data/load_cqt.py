import datasplits
import numpy as np
import os

class Splits():
    train = None
    validation = None
    test = None
    def __init__(self,splits):
        '''
        splits = (train,validation,test)
        train,validation,test are lists of the song names (without file extensions)
        '''
        self.train,self.validation,self.test = splits
        

class load_cqt():
    ''' example usage:
    
    cqt_folderpath = '../../../cqt'
    datasplits_filepath = '../data/datasplits.txt'    
    cqt = load_cqt.load_cqt(cqt_folderpath,datasplits_filepath)   
    
    for song in cqt.splits.train:
        print cqt.load(song).shape
    '''
    
    splits = None # names of songs in train, validation, and test
    
   
    cqt_folderpath = None
    datasplit_filepath = None
    
    def __init__(self,cqt_folderpath,datasplit_filepath='datasplits.txt'):
        self.cqt_folderpath = cqt_folderpath
        self.datasplit_filepath = datasplit_filepath
        
        self.splits = Splits(datasplits.readsplits(path=datasplit_filepath))
    
    def load(self,songname):
        return np.loadtxt(os.path.join(self.cqt_folderpath,songname+'.csv'),delimiter=',')
    