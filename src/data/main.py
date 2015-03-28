# -*- coding: utf-8 -*-
# Author: Junbo (Jake) Zhao

import os
from time import time
from data_loader import *
from data_proc import *
import numpy as np

save_path = '/scratch/jz1672/remeex/features'
os.system('mkdir -p ' + save_path)
melody_type = 2


def read_one_song(data_multi_track):
    melody_anno = read_mdb_melody_anno(data_multi_track, melody_type)
    raw_audio = read_mdb_mix_audio(data_multi_track)
    return (data_multi_track.title, melody_anno, raw_audio)


def mfccs_main():
    print '==> MFCC extracting'
    generator = read_mdb_all_data_generator()
    # Path
    save_path_mfcc = os.path.join(save_path, 'mfcc')
    save_path_anno = os.path.join(save_path, 'melody_type' + str(melody_type))
    os.system('mkdir -p ' + save_path_mfcc)
    os.system('mkdir -p ' + save_path_anno)
    # Extracting
    for g in generator:
        try:
            name, m, r = read_one_song(g)
        except:
            continue
        if not os.path.exists(os.path.join(save_path_mfcc, name+'.csv')):
            mfccs = feature_mfcc_seg(r)
            np.savetxt(os.path.join(save_path_mfcc, name+'.csv'), mfccs, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_anno, name+'.csv')):
            np.savetxt(os.path.join(save_path_anno, name+'.csv'), m, fmt='%.4f', delimiter=',')
        print "==> %s done." % name
    print "==> done!"


def cqt_main():
    print '==> CQT extracting'
    generator = read_mdb_all_data_generator()
    # Path
    save_path_cqt = os.path.join(save_path, 'cqt')
    save_path_anno = os.path.join(save_path, 'melody_type' + str(melody_type))
    os.system('mkdir -p ' + save_path_cqt)
    os.system('mkdir -p ' + save_path_anno)
    # Extracting
    for g in generator:
        try:
            name, m, r = read_one_song(g)
        except:
            continue
        if not os.path.exists(os.path.join(save_path_cqt, name+'.csv')):
            cqt = feature_cqt_seg(r)
            np.savetxt(os.path.join(save_path_cqt, name+'.csv'), cqts, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_anno, name+'.csv')):
            np.savetxt(os.path.join(save_path_anno, name+'.csv'), m, fmt='%.4f', delimiter=',')
        print "==> %s done." % name
    print "==> done!"

if __name__ == '__main__':
    cqt_main()
