# -*- coding: utf-8 -*-
"""DS1003 Final Project
   SVM
   Author: Rita Li
"""


import numpy as np
import pandas as pd
from sklearn import svm, preprocessing

melody_path = '/scratch/jz1672/remeex/features/melody_notes3/'
cqt_path = '/scratch/jz1672/remeex/features/cqt2/'
mfcc_path = '/scratch/jz1672/remeex/features/mfcc/'


datasplits = '/home/ml4713/datasplits.txt'


def data_readin():
    data = pd.read_table(datasplits, sep = ',', header = None, index_col = 0)
    data.columns = ['song']
    tr_name = data.loc['train']
    va_name = data.loc['validation']
    te_name = data.loc['test']


    melody_tr = []
    cqt_tr = []
    mfcc_tr = []
    for file_name in tr_name.values:
      melody_tr.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))
      cqt_tr.append(np.loadtxt(cqt_path + file_name[0] + '.csv', delimiter = ','))
      mfcc_tr.append(np.loadtxt(mfcc_path + file_name[0] + '.csv', delimiter = ','))


    melody_va = []
    cqt_va = []
    mfcc_va = []
    for file_name in va_name.values:
      melody_va.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))
      cqt_va.append(np.loadtxt(cqt_path + file_name[0] + '.csv', delimiter = ','))
      mfcc_va.append(np.loadtxt(mfcc_path + file_name[0] + '.csv', delimiter = ','))


    melody_te = []
    cqt_te = []
    mfcc_te = []
    for file_name in te_name.values:
      melody_te.append(np.loadtxt(melody_path + file_name[0] + '.csv', delimiter = ','))
      cqt_te.append(np.loadtxt(cqt_path + file_name[0] + '.csv', delimiter = ','))
      mfcc_te.append(np.loadtxt(mfcc_path + file_name[0] + '.csv', delimiter = ','))

    return melody_tr, cqt_tr, mfcc_tr, melody_va, cqt_va, mfcc_va, melody_te, cqt_te, mfcc_te


def data_prep(melody, cqt, mfcc):
    y = pd.DataFrame(np.hstack(melody), columns = ['y'])
    cqt = pd.DataFrame(np.vstack(cqt))
    print cqt.shape
    mfcc = pd.DataFrame(np.vstack(mfcc))

    return pd.concat([y, cqt, mfcc], axis = 1)



def basic_model_data():

    mtr, cqt_tr, mfcc_tr, mva, cqt_va, mfcc_va, mte, cqt_te, mfcc_te = data_readin()

    train = data_prep(mtr, cqt_tr, mfcc_tr)
    validation = data_prep(mva, cqt_va, mfcc_va)
    test = data_prep(mte, cqt_te, mfcc_te)

    tr_x = train.drop('y', axis = 1)
    tr_y = train.y

    va_x = validation.drop('y', axis = 1)
    va_y = validation.y

    te_x = test.drop('y', axis = 1)
    te_y = test.y

    return tr_x, tr_y, va_x, va_y, te_x, te_y


def update_weight(err_ind, old_weight):
    indicator = np.zeros(len(old_weight))
    indicator[err_ind] = 1
    new_weight = old_weight.copy()
    err_rate = 1 / sum(old_weight) * np.dot(old_weight, indicator)
    for ind in err_ind:
	  new_weight[ind] = old_weight[ind] * (1 - err_rate) / err_rate
    return new_weight, err_rate



def AdaBoost(num_rounds, train_x, train_y, test_x, test_y):
    weight= np.ones(train_x.shape[0]) / float(train_x.shape[0])
    Gm_tr = np.zeros(num_rounds * train_y.shape[0]).reshape(num_rounds, train_y.shape[0]) #initialize classifier for each sample
    Gm_te = np.zeros(num_rounds * test_y.shape[0]).reshape(num_rounds, test_y.shape[0])
    err = np.zeros(num_rounds)
    final_tr = pd.DataFrame(np.zeros(num_rounds * train_y.shape[0]).reshape(num_rounds, train_y.shape[0]))
    final_te = pd.DataFrame(np.zeros(num_rounds * test_y.shape[0]).reshape(num_rounds, test_y.shape[0]))
    weighted_train_x = train_x.copy(deep = True)
    for i in range(num_rounds):
        clf = svm.LinearSVC(C = 1, penalty = 'l1', dual = False).fit(weighted_train_x, train_y)
        Gm_tr[i, :] = clf.predict(train_x)
        Gm_te[i, :] = clf.predict(test_x)
        err_ind = train_y [train_y != Gm_tr[i, :]].index
        weight, err[i] = update_weight(err_ind, weight)
        final_tr.iloc[i, :] = np.log((1 - err[i]) / err[i]) * Gm_tr[i]
        final_te.iloc[i, :] = np.log((1- err[i]) / err[i]) * Gm_te[i]
        weighted_train_x = weighted_train_x.mul(weight, axis = 0)
        print "%d round completed"%(i)

    return np.sign(final_tr.sum(axis = 0)), np.sign(final_te.sum(axis = 0))

def main():
   tr_x, tr_y, va_x, va_y, te_x, te_y = basic_model_data()
   print "train shape: ", tr_x.shape
   print "validation shape", va_x.shape
   print "test shape", te_x.shape
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
   tr_pred, va_pred = AdaBoost(100, normalized_tr_x, tr_vd, normalized_va_x, va_vd)

   print "Voice Detection..."
   c_seq = [10**i for i in range(-5, 5)]
   accuracy = []
   for c in c_seq:
       clf = svm.LinearSVC(C = 10**(-5), penalty = 'l1', dual = False).fit(normalized_tr_x, tr_vd)
       accuracy.append(clf.score(normalized_va_x, va_vd))

   opt_c_vd = c_seq[accuracy.index(max(accuracy))]
   print "validation score ==> ", max(accuracy)
   print "c value for vocie detection: ", opt_c_vd
   opt_model_vd = svm.LinearSVC(C = 10**(-5), penalty = 'l1', dual = False).fit(normalized_tr_x, tr_y)

   print "testing score ==> ", opt_model_vd.score(normalized_te_x, te_y)
   tr_pred_y = pd.Series(opt_model_vd.predict(normalized_tr_x), index = tr_x.index)
   new_tr_x = normalized_tr_x[tr_pred_y != 0]
   new_tr_y = tr_y[tr_pred_y != 0]

   va_pred_y = pd.Series(opt_model_vd.predict(normalized_va_x), index = va_x.index)
   new_va_x = normalized_va_x[va_pred_y != 0]
   new_va_y = va_y[va_pred_y != 0]

   te_pred_y = pd.Series(opt_model_vd.predict(normalized_te_x), index = te_x.index)
   new_te_x = normalized_te_x[te_pred_y != 0]
   new_te_y = te_y[te_pred_y != 0]

   print "Frequency Estimation..."
   print " -- Perfect Prediction From Stage 1 -- "
   train_x = normalized_tr_x[tr_vd != 0]
   train_y = tr_y[tr_vd != 0]

   valid_x = normalized_va_x[va_vd != 0]
   valid_y = va_y[va_vd != 0]

   test_x = normalized_te_x[te_vd != 0]
   test_y = te_y[te_vd != 0]

   score = []
   for c in c_seq:
       reg = svm.LinearSVR(C = c, loss = 'squared_epsilon_insensitive', dual = False).fit(train_x, train_y)
       score.append(reg.score(valid_x, valid_y))
   opt_c = c_seq[score.index(max(score))]

   print "validation score ==> ", max(score)
   print "c value: ", opt_c

   opt_model = svm.LinearSVR(C = opt_c, loss = 'squared_epsilon_insensitive', dual = False).fit(train_x, train_y)
   print "tesing score ==> ", opt_model.score(test_x, test_y)

   print " -- Continuous stage process --"
   score = []
   for c in c_seq:
     reg = svm.LinearSVR(C = c, loss = 'squared_epsilon_insensitive', dual = False).fit(new_tr_x, new_tr_y)
     score.append(reg.score(new_va_x, new_va_y))

   opt_c = c_seq[score.index(max(score))]

   print "validaiton score ==> ", max(score)
   print "c value: ", opt_c


   opt_model = svm.LinearSVR(C = opt_c, loss = 'squared_epsilon_insensitive', dual = False).fit(new_tr_x, new_tr_y)
   print "testing score ==>  ", opt_model.score(new_te_x, new_te_y)
   print "Notes Class Prediction..."
   score = []
   for c in c_seq:
       multi_cls = svm.LinearSVC(C = c, penalty = 'l1', dual = False).fit(new_tr_x, new_tr_y)
       score.append(multi_cls.score(new_va_x, new_va_y))
   opt_c = c_seq[score.index(max(score))]

   print "validation score: ", max(score)
   print "c value: ", opt_c

   opt_multi_cls = svm.LinearSVC(C = opt_c, penalty = 'l1', dual = False).fit(new_tr_x, new_tr_y)
   print "testing score: ", opt_multi_cls(new_te_x, new_te_y)

if __name__ == '__main__':
  main()
