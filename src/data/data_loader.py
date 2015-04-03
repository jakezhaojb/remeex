# -*- coding: utf-8 -*-
# ****************************************************************************
# Author: Junbo (Jake) Zhao
# Description: This file includes loading functions of audio and annotation.
# ****************************************************************************

from __future__ import division
import os
os.environ['MEDLEYDB_PATH'] = '/misc/vlgscratch2/LecunGroup/jakez/MedleyDB'
import medleydb as mdb
import glob
import types
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from librosa import load


# Three types of melody
melody_type_dict = {
            1: 'melody1_annotation',
            2: 'melody2_annotation',
            3: 'melody3_annotation',
        }

# Note: reading melody must be ahead of reading audio.
# TODO I am just lasy now; will wrap everything in a class... :)
num_segments = 0


def read_mdb_all_data_generator():
    """This function calls for loading mdb data
    Return:
        generator
    """
    dataset = mdb.load_all_multitracks()
    return dataset


def read_mdb_data_generator(tpath):
    """This function calls for loading mdb data
    Args:
        tpath: string; track folder paths
               regex accepted.
    Return:
        generator
    """
    assert isinstance(tpath, str)
    track_list = glob.glob(os.path.join(mdb.AUDIO_DIR), tpath)
    data_subset = mdb.load_multitracks(track_list)
    return data_subset


def read_mdb_melody_anno(data_multi_track, melody_type):
    """This function calls for loading mdb data
    Args:
        data_multi_track: one iterate from mdb data generator
    Return:
        np.array: melody array; shape=[length,]
    """
    assert isinstance(data_multi_track, mdb.multitrack.MultiTrack)
    assert isinstance(melody_type, int)
    try:
        melody_anno_attr = getattr(data_multi_track, melody_type_dict.get(melody_type))
    except:
        raise Exception("This song doesn't have melody annotations")
    melody_anno = map(lambda x: x[1], melody_anno_attr)
    # Not sure about the first and the last hop
    global num_segments
    num_segments = len(melody_anno) - 2
    return np.array(melody_anno[1:-1])


def split_audio(data, num, sample_rate=22050):
    """This function splits raw wave data into several pieces that matches melody annotations.
    Args:
        data: numpy.ndarray; shape=(length,)
    Return:
        numpy.ndarray: (num_segments, length_segments)
    """
    if sample_rate == 22050:
        length_seg = 128
    elif sample_rate == 44100:
        length_seg = 256
    else:
        raise Exception('Abnormal sample rate.')
    # Splitting starts
    # Kick out the first half frame.
    data_clip = data[int(length_seg/2):]
    split_data = np.zeros(shape=[num, length_seg])
    for i in range(num):
        split_data[i,:] = data_clip[i*length_seg:(i+1)*length_seg].reshape(1,length_seg)
    return split_data
    

def read_mdb_mix_audio(data_multi_track, sample_rate=22050, split=True):
    """This function calls for loading mdb data
    Args:
        data_multi_track: one iterate from mdb data generator
    Return:
        np.ndarray: segmented raw audio float number; shape=[num_segments, length_seg]
    """
    assert isinstance(data_multi_track, mdb.multitrack.MultiTrack)
    path_to_wav = data_multi_track.raw_audio[0].mix_path
    raw_data = load(path_to_wav, sr=sample_rate)
    if not split:
        return raw_data[0]
    global num_segments
    assert num_segments > 0 
    split_mix_audio = split_audio(raw_data[0], num_segments)
    num_segments = 0  # safeguard
    return split_mix_audio
