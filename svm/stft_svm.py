# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import svm, preprocessing

melody_path = '/misc/vlgscratch2/LecunGroup/jakez/melody_type1/'
stft_path = '/misc/vlgscratch2/LecunGroup/jakez/stft/'

datasplits = '/home/ml4713/datasplits.txt'


def stft_proc(file_name):
      a = open(file_name).readlines()
      b = map(lambda x: x.replace('(','').replace(')','').split(','), a)
      c = list(b)
      cc = np.array(c)
      d = np.array(list(map(lambda y: list(map(lambda x: abs(complex(x.strip())), y)), cc)))
      return d



def data_readin():
    data = pd.read_table(datasplits, sep = ',', header = None, index_col = 0)
    data.columns = ['song']
    tr_name = data.loc['train']
    va_name = data.loc['validation']
    te_name = data.loc['test']

    melody_tr = []
    stft_tr = []
    for file_name in tr_name.values:
      melody_tr.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))
      stft_tr.append(stft_proc(stft_path + file_name[0] + '.csv'))


    melody_va = []
    stft_va = []
    for file_name in va_name.values:
        melody_va.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))
        stft_va.append(stft_proc(stft_path + file_name[0] + '.csv'))

    melody_te = []
    stft_te = []
    for file_name in te_name.values:
      melody_te.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))
      stft_te.append(stft_proc(stft_path + file_name[0] + '.csv'))


    return  melody_tr, stft_tr, melody_va, stft_va, melody_te, stft_te,

def data_prep(melody, stft):
    y = pd.DataFrame(np.hstack(melody), columns = ['y'])
    stft = pd.DataFrame(np.vstack(stft))
    return pd.concat([y, stft], axis = 1)


def basic_model_data():

    mtr, stft_tr, mva, stft_va, mte, stft_te = data_readin()

    train = data_prep(mtr, stft_tr)
    validation = data_prep(mva, stft_va)
    test = data_prep(mte, stft_te)

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
       clf = svm.LinearSVC(C = 1, penalty = 'l1', dual = False).fit(normalized_tr_x, tr_vd)
       accuracy.append(clf.score(normalized_va_x, va_vd))
   print accuracy
   opt_c_vd = c_seq[accuracy.index(max(accuracy))]
   print "validation score ==> ", max(accuracy)
   print "c value for vocie detection: ", opt_c_vd

   opt_model = svm.LinearSVC(C = opt_c_vd, penalty = 'l1', dual = False).fit(normalized_tr_x, tr_vd)
   print "testing score: ", opt_model(normalized_te_x, te_vd)

if __name__ == '__main__':
  main()
