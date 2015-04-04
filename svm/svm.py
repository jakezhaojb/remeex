# -*- coding: utf-8 -*-
"""DS1003 Final Project
   SVM
   Author: Rita Li
"""
import numpy as np
import pandas as pd
import random
from sklearn import svm
import os

melody_path = '/scratch/jz1672/remeex/features/melody_type1/'
cqt_path = '/scratch/jz1672/remeex/features/cqt2/'
mfcc_path = '/scratch/jz1672/remeex/features/mfcc/'

def data_prep():
    melody = []
    for filename in os.listdir(melody_path):
         melody.append(np.loadtxt(melody_path + filename, delimiter = ','))
    cqt = []
    for filename in os.listdir(cqt_path):
       cqt.append(np.loadtxt(cqt_path + filename, delimiter = ','))
    mfcc = []
    for filename in os.listdir(mfcc_path):
       mfcc.append(np.loadtxt(mfcc_path + filename, delimiter = ','))
    melody = np.hstack(melody)
    cqt = np.vstack(cqt)
    mfcc = np.vstack(mfcc)
    return melody, cqt, mfcc


def basic_model(melody, cqt, mfcc):
    num = melody.shape[0]
    y = pd.DataFrame(melody, columns = ['melody'])
    cqt = pd.DataFrame(cqt)
    mfcc = pd.DataFrame(mfcc)

    medley = pd.concat([cqt, mfcc, y], axis = 1)

    tr_ind = random.sample(set(y.index), int(np.floor(num * 0.7)))
    tr = medley.loc[tr_ind, :]
    te = medley.loc[medley.index - tr_ind, :]

    tr_x = tr.drop('melody', axis = 1)
    tr_y = tr.melody

    te_x = te.drop('melody', axis = 1)
    te_y = te.melody

    return tr_x, tr_y, te_x, te_y


def main():
   mel, cqt, mfcc = data_prep()
   print "Data Loaded"

   tr_x, tr_y, te_x, te_y = basic_model(mel, cqt, mfcc)
   print "Model Building..."

   c_seq = [0.001, 0.01, 0.1, 1, 10, 100]
   score = []
   for c in c_seq:
     clf = svm.LinearSVC(C = c, dual = False, penalty = 'l1').fit(tr_x, tr_y)
     score.append(clf.score(te_x, te_y))

   print "highest score ==> ", max(score)
   print "c value: ", c_seq[score.index(max(score))]



if __name__ == '__main__':
  main()