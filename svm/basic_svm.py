# -*- coding: utf-8 -*-
"""DS1003 Final Project
   SVM
   Author: Rita Li
"""
import numpy as np
import pandas as pd
import random
from sklearn import svm, preprocessing
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


def basic_model_data(melody, cqt, mfcc):
    num = melody.shape[0]
    y = pd.DataFrame(melody, columns = ['melody'])
    cqt = pd.DataFrame(cqt)
    mfcc = pd.DataFrame(mfcc)

    medley = pd.concat([cqt, mfcc, y], axis = 1)


    np.random.seed(52)
    tr_ind = random.sample(set(y.index), int(np.floor(num * 0.5)))
    va_ind = random.sample(set([va_ind for va_ind in set(medley.index) if va_ind not in tr_ind]), int(np.floor((num - len(tr_ind)) * 0.5)))
    te_ind = [te_ind for te_ind in set(medley.index) if te_ind not in tr_ind and te_ind not in va_ind]


    tr = medley.loc[tr_ind, :]
    va = medley.loc[va_ind, :]
    te = medley.loc[te_ind, :]

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

   tr_x, tr_y, va_x, va_y, te_x, te_y = basic_model_data(mel, cqt, mfcc)
   #normalize the data
   scaler_x = preprocessing.StandardScaler().fit(tr_x)
   normalized_tr_x = pd.DataFrame(scaler_x.transform(tr_x), index = tr_x.index)
   normalized_va_x = pd.DataFrame(scaler_x.transform(va_x), index = va_x.index)
   normalized_te_x = pd.DataFrame(scaler_x.transform(te_x), index = te_x.index)
   #binary labels
   tr_vd = tr_y.copy(deep = True)
   va_vd = va_y.copy(deep = True)
   te_vd = te_y.copy(deep = True)
   tr_vd.values[tr_vd.values != 0] = 1
   va_vd.values[va_vd.values != 0] = 1
   te_vd.values[te_vd.values != 0] = 1
   print "Model Building..."
   print "Voice Detection..."
   c_seq = [10**i for i in range(-3, 4)]
   accuracy = []
   for c in c_seq:
       clf = svm.LinearSVC(C = c, penalty = 'l1', dual = False).fit(normalized_tr_x, tr_vd)
       accuracy.append(clf.score(normalized_va_x, va_vd))

   opt_c_vd = c_seq[accuracy.index(max(accuracy))]
   print "validation score ==> ", max(accuracy)
   print "c value for vocie detection: ", opt_c_vd


   opt_model_vd = svm.LinearSVC(C = opt_c_vd, penalty = 'l1', dual = False).fit(normalized_tr_x, tr_vd)
   tr_pred_y = pd.Series(opt_model_vd.predict(normalized_tr_x), index = tr_x.index)
   new_tr_x = normalized_tr_x[tr_pred_y != 0]
   new_tr_y = tr_y[tr_pred_y != 0]
   
   scaler_y = preprocessing.StandardScaler().fit(new_tr_y)
   new_tr_y = pd.Series(scaler_y.transform(new_tr_y), index = new_tr_y)


   va_pred_y = pd.Series(opt_model_vd.predict(normalized_va_x), index = va_x.index)
   new_va_x = normalized_va_x[va_pred_y != 0]
   new_va_y = normalized_va_y[va_pred_y != 0]
   new_va_y = pd.Series(scaler_y.transform(new_va_y), index = new_va_y)
   

   te_pred_y = pd.Series(opt_model_vd.predict(normalized_te_x), index = te_x.index)
   new_te_x = normalized_te_x[te_pred_y != 0]
   new_te_y = normalized_te_y[te_pred_y != 0]
   new_te_y = pd.Series(scaler_y.transform(new_te_y), index = new_te_y)
   print "Frequency Estimation..."
   print " -- Continuous stage process --"
   c_seq = [10**i for i in range(-3, 4)]
   score = []
   for c in c_seq:
     clf = svm.LinearSVR(C = c, loss = 'squared_epsilon_insensitive', dual = False).fit(new_tr_x, new_tr_y)
     score.append(clf.score(new_va_x, new_va_y))

   opt_c = c_seq[score.index(max(score))]

   print "validaiton score ==> ", max(score)
   print "c value: ", opt_c


   opt_model = svm.LinearSVR(C = opt_c, loss = 'squared_epsilon_insensitive', dual = False).fit(new_tr_x, new_tr_y)
   print "testing score ==>  ", opt_model.score(new_te_x, new_te_y)


if __name__ == '__main__':
  main()
