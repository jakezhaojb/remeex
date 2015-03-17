# -*- coding: utf-8 -*-
# Junbo (Jake) Zhao


import sys
import os
from scipy import savemat
import cPickle


def main(argv):
    # TODO unit test it
    assert len(argv) == 2
    path = argv[1]
    assert os.path.exists(path)
    dir_path = os.path.abspath(os.path.dirname(path))
    tmp_mat_path = os.path.join(dir_path, 'tmp.mat')  # TODO modify the temp
    # load literal file
    file = open(path, 'r')
    data = cPickle.load(file)
    file.close()
    savemat(data, tmp_mat_path)


if __name__ == '__main__':
    main(sys.argv)
