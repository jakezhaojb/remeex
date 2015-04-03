# -*- coding: utf-8 -*-
# Jake

import sys
import numpy as np
from scipy import io


def combine(data_file, target_file):
    data = np.loadtxt(data_file, delimiter=',')
    target = np.loadtxt(target_file, delimiter=',')
    dict_save = {}
    # some praperation of coherence
    if len(data.shape) == 1:
        data = data.reshape(data.size, 1)
    if len(target.shape) == 1:
        target = target.reshape(target.size, 1)
    dict_save['1'] = data.T
    dict_save['2'] = target.T
    return dict_save


def save_to_mat(dict, path):
    assert isinstance(path, str)
    if not path.endswith('.mat'):
        path += '.mat'
    io.savemat(path, dict)


def main(argv):
    if not len(argv) == 4:
        print 'Usage: \n\t python csv2mat.py datafile.csv targetfile.csv dstfile.mat \n'
        sys.exit()
    src1, src2, dst = argv[1], argv[2], argv[3]
    dict = combine(src1, src2)
    save_to_mat(dict, dst)


if __name__ == '__main__':
    main(sys.argv)
