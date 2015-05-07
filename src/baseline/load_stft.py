# -*- coding: utf-8 -*-
# Jake

import os
os.environ['MEDLEYDB_PATH']='/misc/vlgscratch2/LecunGroup/jakez/MedleyDB'
import sys
sys.path.append('/home/jz1672/lib/python2.7/site-packages/medleydb-0.1.0-py2.7.egg')
import numpy as np

g_stft_path = '/scratch/jz1672/remeex/features/stft'
g_save_path = '/scratch/jz1672/remeex/stft_pred'
g_melody_path = '/scratch/jz1672/remeex/features/melody_type1'
os.system('mkdir -p ' + g_save_path)


def load_stft(file_name):
    a = open(file_name).readlines()
    b = map(lambda x: x.replace('(','').replace(')','').split(','), a)
    d = np.array(map(lambda y: list(map(lambda x: abs(complex(x)), y)), np.array(b)))
    e = d.argmax(1) * 22050 / 1024
    return e


def compare(arr1, arr2):
    assert len(arr1) == len(arr2)
    return len(arr1), np.sum(arr1 == arr2)


def process():
    assert os.path.isdir(g_stft_path)
    length, acc = 0, 0
    for fl in os.listdir(g_stft_path):
        melody_arr = np.loadtxt(os.path.join(g_melody_path, fl))
        melody_arr = np.piecewise(melody_arr, [melody_arr > 0], [1])
        # stft
        abs_fl = os.path.join(g_stft_path, fl)
        lists = load_stft(abs_fl)
        lists = np.piecewise(lists, [lists > 0], [1])
        # Compare
        length_fl, acc_fl = compare(lists, melody_arr)
        length += length_fl
        acc += acc_fl
        print 'Done: {}'.format(fl)


if __name__ == '__main__':
    process()
