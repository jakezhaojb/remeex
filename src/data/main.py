# -*- coding: utf-8 -*-
# Author: Junbo (Jake) Zhao

import os
from time import time
from data_loader import *
from data_proc import *
import numpy as np
import math

save_path = '/scratch/jz1672/remeex/features'
os.system('mkdir -p ' + save_path)
melody_type = 3


def read_one_song(data_multi_track):
    melody_anno = read_mdb_melody_anno(data_multi_track, melody_type)
    raw_audio = read_mdb_mix_audio(data_multi_track, sample_rate=22050, split=False)
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


def cqt_main_seg():
    print '==> CQT extracting'
    generator = read_mdb_all_data_generator()
    # Path
    save_path_cqt = os.path.join(save_path, 'cqt')
    save_path_anno = os.path.join(save_path, 'melody_type' + str(melody_type))
    save_path_anno_notes = os.path.join(save_path, 'melody_notes')
    os.system('mkdir -p ' + save_path_cqt)
    os.system('mkdir -p ' + save_path_anno)
    os.system('mkdir -p ' + save_path_anno_notes)
    # Extracting
    for g in generator:
        try:
            name, m, r = read_one_song(g)
            notes = melody_to_midi(m,fmin=32.7032)
        except:
            continue
        if not os.path.exists(os.path.join(save_path_cqt, name+'.csv')):
            cqt = feature_cqt_seg(r)
            np.savetxt(os.path.join(save_path_cqt, name+'.csv'), cqts, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_anno, name+'.csv')):
            np.savetxt(os.path.join(save_path_anno, name+'.csv'), m, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_anno_notes, name+'.csv')):
            np.savetxt(os.path.join(save_path_anno_notes, name+'.csv'), notes, fmt='%d', delimiter=',')
        print "==> %s done." % name
    print "==> done!"


def cqt_main_whole():
    print '==> CQT extracting'
    generator = read_mdb_all_data_generator()
    # Path
    save_path_cqt = os.path.join(save_path, 'cqt2')
    save_path_anno = os.path.join(save_path, 'melody_type' + str(melody_type))
    save_path_anno_notes = os.path.join(save_path, 'melody_notes' + str(melody_type))
    os.system('mkdir -p ' + save_path_cqt)
    os.system('mkdir -p ' + save_path_anno)
    os.system('mkdir -p ' + save_path_anno_notes)
    # Extracting
    for g in generator:
        try:
            name, m, r = read_one_song(g)
            notes = melody_to_midi(m, fmin=32.7032)
        except:
            continue
        if not os.path.exists(os.path.join(save_path_cqt, name+'.csv')):
            cqts = feature_cqt_whole(r)
            np.savetxt(os.path.join(save_path_cqt, name+'.csv'), cqts, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_anno, name+'.csv')):
            np.savetxt(os.path.join(save_path_anno, name+'.csv'), m, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_anno_notes, name+'.csv')):
            np.savetxt(os.path.join(save_path_anno_notes, name+'.csv'), notes, fmt='%d', delimiter=',')
        print "==> %s done." % name
    print "==> done!"


def raw_main():
    print "==> Raw extracting"
    generator = read_mdb_all_data_generator()
    # Path
    save_path_raw = os.path.join(save_path, 'raw')
    save_path_anno = os.path.join(save_path, 'melody_type' + str(melody_type))
    save_path_anno_notes = os.path.join(save_path, 'melody_notes' + str(melody_type))
    os.system('mkdir -p ' + save_path_raw)
    os.system('mkdir -p ' + save_path_anno)
    os.system('mkdir -p ' + save_path_anno_notes)
    # Extracting
    for g in generator:
        try:
            name, m, r = read_one_song(g)
            notes = melody_to_midi(m, fmin=32.7032)
        except:
            continue
        if not os.path.exists(os.path.join(save_path_raw, name+'.csv')):
            np.savetxt(os.path.join(save_path_raw, name+'.csv'), r, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_anno, name+'.csv')):
            np.savetxt(os.path.join(save_path_anno, name+'.csv'), m, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_anno_notes, name+'.csv')):
            np.savetxt(os.path.join(save_path_anno_notes, name+'.csv'), notes, fmt='%d', delimiter=',')
        print "==> %s done." % name
    print "==> done!"


def raw_main_seg():
    print "==> Segmented Raw extracting"
    generator = read_mdb_all_data_generator()
    # Path
    save_path_raw = os.path.join(save_path, 'raw_seg')
    os.system('mkdir -p ' + save_path_raw)
    # Extracting
    for g in generator:
        try:
            name, m, r = read_one_song(g)
        except:
            continue
        r_seg = split_audio(r, m.shape[0])
        if not os.path.exists(os.path.join(save_path_raw, name+'.csv')):
            np.savetxt(os.path.join(save_path_raw, name+'.csv'), r_seg, fmt='%.4f', delimiter=',')
        print "==> %s done." % name
    print "==> done!"


def raw_main_seg_stride(stride=4096):
    print "==> Segmented Raw (with stride) extracting"
    generator = read_mdb_all_data_generator()
    # Path
    save_path_raw = os.path.join(save_path, 'raw_seg_stride_' + str(stride))
    save_path_label = os.path.join(save_path, 'raw_seg_stride_' + str(stride) + '_label')
    os.system('mkdir -p ' + save_path_raw)
    os.system('mkdir -p ' + save_path_label)
    # for sample_rate = 22050
    assert stride % 128 == 0
    for g in generator:
        try:
            name, m, r = read_one_song(g)
        except:
            continue
        # for sample_rate = 22050
        offset = 64
        offset_label = 0
        length = int(math.floor((len(r)-64-22016)/stride)) # +1
        r_seg_stride = np.zeros((length, 22016))
        m_seg_stride = np.zeros((length, 172))
        for i in range(length):
            r_seg_stride[i, :] = r[offset: offset + 22016]  # To match the hop sizes
            m_seg_stride[i, :] = m[offset_label: offset_label+172]
            offset += stride
            offset_label += stride / 128
        if not os.path.exists(os.path.join(save_path_raw, name+'.csv')):
            np.savetxt(os.path.join(save_path_raw, name+'.csv'), r_seg_stride, fmt='%.4f', delimiter=',')
        if not os.path.exists(os.path.join(save_path_label, name+'.csv')):
            np.savetxt(os.path.join(save_path_label, name+'.csv'), m_seg_stride, fmt='%.4f', delimiter=',')
        print "==> %s done." % name
    print '==> done.'


def raw_main_whole():
    print "==> Whole Raw extracting"
    generator = read_mdb_all_data_generator()
    # Path
    save_path_raw = os.path.join(save_path, 'raw_whole')
    os.system('mkdir -p ' + save_path_raw)
    # Extracting
    for g in generator:
        try:
            name, m, r = read_one_song(g)
        except:
            continue
        # for sample_rate = 22050
        r_whole = r[64: 64 + m.shape[0]*128]
        if not os.path.exists(os.path.join(save_path_raw, name+'.csv')):
            np.savetxt(os.path.join(save_path_raw, name+'.csv'), r_whole, fmt='%.4f', delimiter=',')
        print "==> %s done." % name
    print "==> done!"


if __name__ == '__main__':
    raw_main_seg_stride()
