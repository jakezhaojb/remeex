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

melody_path = '/Users/ritali/desktop/ds1003/final_proj/dataset/melody/'
cqt_path = '/Users/ritali/desktop/ds1003/final_proj/dataset/cqt/'
mfcc_path = '/Users/ritali/desktop/ds1003/final_proj/dataset/mfcc/'

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


    np.random.seed(52)
    tr_ind = random.sample(set(y.index), int(np.floor(num * 0.5)))
    va_ind = random.sample(set(medley.index - tr_ind), int(np.floor((num - len(tr_ind)) * 0.5)))
    tr = medley.loc[tr_ind, :]
    va = medley.loc[va_ind, :]
    te = medley.loc[medley.index - tr_ind - va_ind, :]

    tr_x = tr.drop('melody', axis = 1)
    tr_y = tr.melody

    va_x = va.drop('melody', axis = 1)
    va_y = va.melody

    te_x = te.drop('melody', axis = 1)
    te_y = te.melody

    return tr_x, tr_y, va_x, va_y, te_x, te_y


def main():
   mel, cqt, mfcc = data_prep()
   print "Data Loaded"

   tr_x, tr_y, va_x, va_y, te_x, te_y = basic_model(mel, cqt, mfcc)
   print "Model Building..."

   c_seq = [0.001, 0.01, 0.1, 1, 10, 100]
   score = []
   for c in c_seq:
     clf = svm.LinearSVR(C = c, loss = 'epsilon_insensitive', dual = False).fit(tr_x, tr_y)
     score.append(clf.score(va_x, va_y))

   opt_c = c_seq[score.index(max(score))]

   print "validaiton score ==> ", max(score)
   print "c value: ", opt_c


   opt_model = svm.LinearSVR(C = opt_c, loss = 'epsilon_insensitive', dual = False).fit(tr_x, tr_y)
   print "testing score ==>  ", opt_model.score(te_x, te_y)


if __name__ == '__main__':
  main()
