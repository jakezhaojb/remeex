import os
import numpy as np
import random
import pandas as pd
import csv


cqt_path = '../../../cqt_44k_hop256_bin84_oct12'




def readsplits(path = 'datasplits.txt'):    
    '''
        returns 3 lists: train,validation,test
        each list contains the names of songs
    '''
    splits = pd.read_csv(path,header=None)
    splits.columns = ['split','song_name']
    
    train = splits.song_name[splits.split=='train'].tolist()
    validation = splits.song_name[splits.split=='validation'].tolist()
    test = splits.song_name[splits.split=='test'].tolist()
    
    return train,validation,test


def generatesplits(path):    
    random.seed(13)
    np.random.seed(13)
    
    song_list = [s.split('.')[0] for s in os.listdir(path)]
    
    num_songs = len(song_list)
    
    shuffled = np.arange(num_songs)
    
    np.random.shuffle(shuffled)
    
    train_split = 0.50
    validation_split = 0.25
    #test_split = 0.25
    
    train_len = int(np.round(num_songs*train_split))
    validation_len = int(np.round(num_songs*validation_split))
    #test_len = int(num_songs - train_len - validation_len)
    
    train_idx = shuffled[:train_len]
    validation_idx = shuffled[train_len:train_len+validation_len]
    test_idx = shuffled[train_len+validation_len:]
    
    train = []
    validation = []
    test = []
    
    for i in train_idx:
        train.append(song_list[i])
        
    for i in validation_idx:
        validation.append(song_list[i])
        
    for i in test_idx:
        test.append(song_list[i])
    
    
    for tr in train:
        if tr in validation:
            print "DUPLICATE\n",tr
        if tr in test:
            print "DUPLICATE\n",tr
    
    for va in validation:
        if va in test:
            print "DUPLICATE\n",va
    
    with open('datasplits.txt','wb') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        for tr in train:
            writer.writerow(['train',tr])
        for va in validation:
            writer.writerow(['validation',va])
        for te in test:
            writer.writerow(['test',te])


