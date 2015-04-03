#-*- coding: utf-8 -*-
# Jake

'''
This file provides aux functions to combine the data samples to torch
'''

import numpy as np
import os
from scipy import io

TARGET_FOLDER = '/scratch/jz1672/remeex/features/melody_type1'
DATA_FOLDER = '/scratch/jz1672/remeex/features/raw_seg'


def load_and_save_to_mat():
    filename_data = os.listdir(DATA_FOLDER)
    filename_target = os.listdir(TARGET_FOLDER)
    for i, j in zip(filename_data, filename_target):
        assert i == j
        filename_data_elem = os.join(DATA_FOLDER, i)
        filename_targer_elem = os.join(TARGET_FOLDER, j)
        data_elem = np.loadtxt(filename_data_elem, delimiter=',')
        target_elem = np.loadtxt(filename_target_elem, delimiter=',')
        # Take a transpose, for mattorch
        data_elem = data_elem.T
        target_elem = target_elem.T
        # 
        dict_save = {}
        dict_save[1] = data_elem
        dict_save[2] = target_elem
        # Filename TODO
        io.savemat('', dict_save)
        
