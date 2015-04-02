# -*- coding: utf-8 -*-
# ****************************************************************************
# Author:
# Description: This file includes basic audio processing.
# Usages: dpark installation is recommended for acceleration.
# ****************************************************************************

# TODO How to check whether dpark is installed?
# TODO How to control the number of RDD made, in makeRDD()?
import librosa
import numpy as np


def feature_logfsgram(y, sr=22050):
    """This function calls for obtaining logfsgram
    Args:
        y (array): time series raw audio data
        sr (int): sample rate
    Return:
        np.array: Log-Frequency-gram
    """
    S_log = librosa.feature.logfsgram(y=y, sr=sr)
    return S_log


def feature_melspectrogram(y, sr=22050):
    """This function calls for obtaining mel-scaled spectrogram
    Args:
        y (array): time series raw audio data
        sr (int): sample rate
    Return:
        np.array: Mel scaled spectrogram
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    return S


def feature_mfcc(y, sr=22050):
    """This function calls for obtaining MFCC
    Args:
        y (array): time series raw audio data
        sr (int): sample rate
    Return:
        np.array: MFCCs
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = mfccs.reshape(mfccs.size,)
    return mfccs


def feature_cqt(y, sr=22050):
    """This function calls for obtaining MFCC
    Args:
        y (array): time series raw audio data
        sr (int): sample rate
    Return:
        np.array: CQTs
    """
    cqts = librosa.cqt(y=y, sr=sr)
    cqts = cqts.reshape(cqts.size,)
    return cqts


def feature_logfsgram_aggr(dpark, y_aggr):
    """This function calls for parallelizing logfsgram calculations
    Args:
        y_aggr (list): list of raw time series raw audio data
    Return:
        list: list of logfsgrams
    """
    assert isinstance(y_aggr, list)
    S_log_aggr = dpark.makeRDD(
                        y_aggr
                        ).map(
                        feature_logfsgram
                        ).collect()
    return S_log_aggr


def feature_melspectrogram_aggr(dpark, y_aggr):
    """This function calls for parallelizing mel-scaled-spectrogram calculations
    Args:
        y_aggr (list): list of raw time series raw audio data
    Return:
        list: list of mel-scaled-spectorgrams
    """
    assert isinstance(y_aggr, list)
    S_aggr = dpark.makeRDD(
                    y_aggr
                    ).map(
                    feature_melspectrogra
                    ).collect()
    return S_aggr


def feature_mfcc_aggr(dpark, y_aggr):
    """This function calls for parallelizing mfccs calculations
    Args:
        y_aggr (list): list of raw time series raw audio data
    Return:
        list: list of mfccs
    """
    assert isinstance(y_aggr, list)
    mfccs_aggr = dpark.makeRDD(
                            y_aggr
                            ).map(
                            feature_mfcc
                            ).collect()
    return mfccs_aggr


def feature_mfcc_seg(y, sr=22050):
    """This function calls for obtaining mfcc from segmented raw audios
    Args:
        y (np.ndarray): segmented raw audio data; shape=[num_segments, length_segment]
        sr (int): sample rate
    Return:
        np.ndarray: MFCCs on segmented raw audios; shape=[num_segments,length_mfccs]
    """
    assert isinstance(y, np.ndarray)
    length_mfccs = feature_mfcc(y[0,:], sr).size
    mfccs = np.zeros(shape=[y.shape[0], length_mfccs])
    for row in range(y.shape[0]):
        mfccs[row,:] = feature_mfcc(y[row,:], sr)
    return mfccs
        

def feature_cqt_seg(y, sr=22050):
    """This function calls for obtaining mfcc from segmented raw audios
    Args:
        y (np.ndarray): segmented raw audio data; shape=[num_segments, length_segment]
        sr (int): sample rate
    Return:
        np.ndarray: CQTs on segmented raw audios; shape=[num_segments,length_mfccs]
    """
    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 2
    length_cqts = feature_cqt(y[0,:], sr).size
    cqts = np.zeros(shape=[y.shape[0], length_cqts])
    for row in range(y.shape[0]):
        cqts[row,:] = feature_cqt(y[row,:], sr)
    return cqts


def feature_cqt_whole(y, sr=22050):
    """This function calls for obtaining mfcc from segmented raw audios
    Args:
        y (np.ndarray): segmented raw audio data; shape=[num_segments, length_segment]
        sr (int): sample rate
    Return:
        np.ndarray: CQTs on segmented raw audios; shape=[num_segments,length_mfccs]
    """
    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1
    if sr == 22050:
        length_seg = 128
    elif sr == 44100:
        length_seg = 256
    else:
        raise Exception('Abnormal sample rate.')
    y = y[length_seg:]  # Important
    cqts = librosa.cqt(y, sr=sr, hop_length=length_seg)
    return cqts[:, :-1].T


def melody_to_midi(melody,fmin=None):
    '''Thus function converts melody frequencies to midi integers
    Args:
        melody (np.ndarray): nd array of frequencies in hertz (zeros set back to zero after transform)
        fmin (float): normalize output to this minimum frequency
    Return:
        np.ndarray: integer array of midi notes (rounded to nearest note)
    '''
    midi = librosa.hz_to_midi(melody)
    if fmin != None:
        midi = midi - librosa.hz_to_midi(fmin)
    midi[melody == 0] = 0
    return np.round(midi).astype('int')
