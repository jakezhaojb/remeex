# -*- coding: utf-8 -*-
"""DS1003 Final Project
   SVM
   Author: Rita Li
"""


import numpy as np
import pandas as pd
from sklearn import svm, preprocessing

melody_path = '/scratch/jz1672/remeex/features/melody_type1/'
cqt_path = '/scratch/jz1672/remeex/features/cqt2/'
mfcc_path = '/scratch/jz1672/remeex/features/mfcc/'
raw_path = '/scratch/jz1672/remeex/features/raw_seg/'


datasplits = '/home/ml4713/datasplits.txt'


def data_readin():
    data = pd.read_table(datasplits, sep = ',', header = None, index_col = 0)
    data.columns = ['song']
    tr_name = data.loc['train']
    va_name = data.loc['validation']
    te_name = data.loc['test']


    melody_tr = []
    raw_tr = []
    for file_name in tr_name.values:
      melody_tr.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))
      raw_tr.append(np.loadtxt(raw_path + file_name[0] + '.csv', delimiter = ','))


    melody_va = []
    raw_va = []
    for file_name in va_name.values:
      melody_va.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))

      raw_va.append(np.loadtxt(raw_path + file_name[0] + '.csv', delimiter = ','))


    melody_te = []
    raw_te = []
    for file_name in te_name.values:
      melody_te.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))
      raw_te.append(np.loadtxt(raw_path + file_name[0] + '.csv', delimiter = ','))

    return melody_tr, raw_tr, melody_va, raw_va, melody_te, raw_te


def data_prep(melody, raw):
    y = pd.DataFrame(np.hstack(melody), columns = ['y'])
    raw = pd.DataFrame(np.vstack(raw))

    return pd.concat([y, raw], axis = 1)



def basic_model_data():

    mtr, raw_tr, mva, raw_va, mte, raw_te = data_readin()

    train = data_prep(mtr, raw_tr)
    validation = data_prep(mva, raw_va)
    test = data_prep(mte, raw_te)

    tr_x = train.drop('y', axis = 1)
    tr_y = train.y

    va_x = validation.drop('y', axis = 1)
    va_y = validation.y

    te_x = test.drop('y', axis = 1)
    te_y = test.y

    return tr_x, tr_y, va_x, va_y, te_x, te_y



def main():
   tr_x, tr_y, va_x, va_y, te_x, te_y = basic_model_data()
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

   c_seq = [10**i for i in range(-5, 5)]
   accuracy = []
   for c in c_seq:
       clf = svm.LinearSVC(C = c, penalty = 'l1', dual = False).fit(normalized_tr_x, tr_vd)
       accuracy.append(clf.score(normalized_va_x, va_vd))

   opt_c_vd = c_seq[accuracy.index(max(accuracy))]

   print "validation score ==> ", max(accuracy)
   print "c value for vocie detection: ", opt_c_vd

   opt_model = svm.LinearSVC(C = opt_c_vd, penalty = 'l1', dual = False).fit(normalized_tr_x, tr_vd)
   print "testing score: ", opt_model(normalized_te_x, te_vd)

if __name__ == '__main__':
  main()
