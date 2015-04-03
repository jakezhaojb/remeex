import datasplits
import data_proc
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
        

class load_melodies():
    ''' example usage:
    
    melodies_folderpath = '../../../melody_type1'
    datasplits_filepath = '../data/datasplits.txt'    
    melodies = load_melodies.load_melodies(melodies_folderpath,datasplits_filepath)   
    
    train = melodies.train
    validation = melodies.validation
    test = melodies.test
    '''
    
    splits = None # names of songs in train, validation, and test
    
    train = None # concatenated melodies
    validation = None
    test = None
    
    train_list = None # list of melody arrays
    validation_list = None  # list of melody arrays
    test_list = None  # list of melody arrays
    melody_folderpath = None
    datasplit_filepath = None
    
    def __init__(self,melody_folderpath,datasplit_filepath='datasplits.txt'):
        self.melody_folderpath = melody_folderpath
        self.datasplit_filepath = datasplit_filepath
        
        self.splits = Splits(datasplits.readsplits(path=datasplit_filepath))
        
        self.train, self.train_list = self.loadsplit(self.splits.train)
        self.validation, self.validation_list = self.loadsplit(self.splits.validation)
        self.test, self.test_list = self.loadsplit(self.splits.test)

        
    def loadsplit(self,split,convert_to_notes=True,fmin=32.7023):
        '''
        for each filename in split, load all melodies, convert to notes,
        and concatenate into one giant array.  Also return the indices
        of the array that correspond to the length of each song.
        '''
        
        melody_list = []
        
        for idx,songname in enumerate(split):            
            melody_list.append(np.loadtxt(os.path.join(self.melody_folderpath,songname+'.csv'),delimiter=','))
            
        melodies = np.hstack(melody_list)
        
        if convert_to_notes:
            melodies = data_proc.melody_to_midi(melodies,fmin=fmin)
            
        return melodies,melody_list
        
    