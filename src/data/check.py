# -*- coding: utf-8 -*-
# Jake

''' This file involves some functions to check the correctness of feature extractions. '''

import numpy as np
import os

REMEEX_ROOT = '/scratch/jz1672/remeex/features'
FOLDER_MELODY_TYPE_1 = 'melody_type1'
FOLDER_MELODY_TYPE_2 = 'melody_type2'
FOLDER_MELODY_TYPE_3 = 'melody_type3'
FOLDER_CQT = 'cqt'
FOLDER_CQT2 = 'cqt2'
FOLDER_MFCC = 'mfcc'


def check_dimension(folder_1, folder_2):
    filename_1 = os.listdir(os.path.join(REMEEX_ROOT, folder_1))
    filename_2 = os.listdir(os.path.join(REMEEX_ROOT, folder_2))
    assert len(filename_1) == len(filename_2)
    for i, j in zip(filename_1, filename_2):
        elem1 = np.loadtxt(os.path.join(REMEEX_ROOT, folder_1, i), delimiter=',')
        elem2 = np.loadtxt(os.path.join(REMEEX_ROOT, folder_2, j), delimiter=',')
        '''
        if elem1.shape[0] - 1 == elem2.shape[0]:
            print 'modify %s' % i
            elem1 = elem1[:-1]
            np.savetxt(os.path.join(REMEEX_ROOT, folder_1, i), elem1, delimiter=',')
        '''
        assert elem1.shape[0] == elem2.shape[0]
    print 'No assertion violated.'


def main():
    #check_dimension(FOLDER_MELODY_TYPE_1, FOLDER_MFCC)
    #check_dimension(FOLDER_MELODY_TYPE_2, FOLDER_MFCC)
    #check_dimension(FOLDER_MELODY_TYPE_3, FOLDER_MFCC)
    check_dimension(FOLDER_CQT2, FOLDER_MFCC)

if __name__ == '__main__':
    main()
