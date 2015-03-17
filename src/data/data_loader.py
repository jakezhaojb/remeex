# -*- coding: utf-8 -*-
# ****************************************************************************
# Author: Junbo (Jake) Zhao
# Description: This file includes loading functions of audio and annotation.
# ****************************************************************************

from __future__ import division
import medleydb as mdb
import os
import glob
import types
from librosa import load


def read_mdb_data_generator(tpath):
    """This function calls for loading mdb data
    Args:
        tpath: string; track folder paths
               regex accepted.
    Return:
        tuple: consist of two elements.
               (length, generator)
    """
    assert isinstance(tpath, str)
    track_list = glob.glob(os.path.join(mdb.AUDIO_DIR), tpath)
    data_length = len(track_list)
    data_subset = mdb.load_multitracks(track_list)
    return (data_length, data_subset)


def read_mdb_melody_anno(data_multi_track):
    """This function calls for loading mdb data
    Args:
        data_multi_track: one iterate from mdb data generator
    Return:
        dict: keys are 'melody1', 'melody2', 'melody3';
              values are 2-D matrices of melody annotations from tracks
    """
    # TODO is the 3 fixed?
    assert isinstance(data_multi_track, mdb.multitrack.MultiTrack)
    k = ['melody1', 'melody2', 'melody3']
    v = [data_multi_track.melody1_annotation,
         data_multi_track.melody2_annotation,
         data_multi_track.melody3_annotation]
    return dict(zip(k, v))


def read_mdb_melody_anno_aggr(data_tuple):
    """This function calls for loading mdb data
    Args:
        data_tuple: (length, data_subset_generator)
    Return:
        list: listing melody annotation dictionaries.
    """
    # TODO this can be better.
    assert isinstance(data_tuple, tuple)
    length = data_tuple[0]
    data_subset_generator = data_tuple[1]
    assert isinstance(data_subset_generator, types.GeneratorType)
    melody_anno = []
    for _ in range(length):
        elem = next(data_subset_generator)
        melody_anno.append(read_mdb_melody_anno(elem))
    return melody_anno


def read_mdb_mix_audio(data_multi_track, sample_rate=22050):
    """This function calls for loading mdb data
    Args:
        data_multi_track: one iterate from mdb data generator
    Return:
        np.array: raw wave float numbers
    """
    assert isinstance(data_multi_track, mdb.multitrack.MultiTrack)
    path_to_wav = data_multi_track.raw_audio[0].mix_path
    raw_data = load(path_to_wav, sr=sample_rate)
    return raw_data


def read_mdb_mix_audio_aggr(data_tuple, sample_rate=22050):
    """This function calls for loading mdb data
    Args:
        data_tuple: (length, data_subset_generator)
    Return:
        list: listing raw wave data as np.array
    """
    # TODO this can be better.
    assert isinstance(data_tuple, tuple)
    length = data_tuple[0]
    data_subset_generator = data_tuple[1]
    assert isinstance(data_subset_generator, types.GeneratorType)
    raw_audio = []
    for _ in range(length):
        elem = next(data_subset_generator)
        raw_audio.append(read_mdb_mix_audio(elem, sample_rate))
    return raw_audio
