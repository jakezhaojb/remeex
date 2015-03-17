# -*- coding: utf-8 -*-
# ****************************************************************************
# Author:
# Description: This file includes basic audio processing.
# Usages: dpark installation is recommended for acceleration.
# ****************************************************************************

# TODO How to check whether dpark is installed
import dpark
import librosa


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
    return mfccs


def feature_logfsgram_aggr(y_aggr):
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


def feature_melspectrogram_aggr(y_aggr):
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


def feature_mfcc_aggr(y_aggr):
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
