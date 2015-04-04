# -*- coding: utf-8 -*-
# Jake

''' Main file '''

import os
import sys
import numpy as np
from scipy import io
from csv2mat import save_to_mat


def combine_save_aggr(data_file, target_file, folder_name, aggr_num=172):
    assert isinstance(aggr_num, int)
    assert isinstance(folder_name, str)
    os.system('mkdir -p ' + folder_name)
    data = np.loadtxt(data_file, delimiter=',')
    target = np.loadtxt(target_file, delimiter=',')
    dict_save = {}
    # some praperation of coherence
    if len(data.shape) == 1:
        data = data.reshape(data.size, 1)
    if len(target.shape) == 1:
        target = target.reshape(target.size, 1)
    # Start aggregating
    # Sample_rate = 22050
    assert data.shape[0] != 1 and data.shape[1] != 1  # This scipt is running on the segmented raw audios
    num = data.shape[0] / aggr_num
    for i in range(num):
        filename_mat = os.path.join(folder_name, str(i)+'.mat')
        filename_t7b = os.path.join(folder_name, str(i)+'.t7b')
        dict_save = {}
        # Aggregate
        data_elem = data[i*aggr_num:(i+1)*aggr_num, :]
        target_elem = target[i*aggr_num:(i+1)*aggr_num, :]
        dict_save['1'] = data_elem.reshape(data_elem.size, 1)
        dict_save['2'] = target_elem.reshape(target_elem.size, 1)
        save_to_mat(dict_save, filename_mat)
        os.system('th mat2t7b.lua --src ' + filename_mat + ' --dst ' + filename_t7b)
        os.system('rm ' + filename_mat)


def combine_save_aggr_main(data_folder, target_folder, folder_name):
    data_file_list = os.listdir(data_folder)
    target_file_list = os.listdir(target_folder)
    if os.path.exists(folder_name):
        print '%s already exists.' % folder_name
        return
    def iter(args):
        data_file = args[0]
        target_file = args[1]
        assert data_file == target_file
        subfolder_name = os.path.join(folder_name, data_file)
        data_file_complete = os.path.join(data_folder, data_file)
        target_file_complete = os.path.join(target_folder, target_file)
        combine_save_aggr(data_file_complete, target_file_complete, subfolder_name)
        print "%s done." % data_file
    map(iter, zip(data_file_list, target_file_list))


def main(argv):
    if not len(argv) == 4:
        print 'Usage: \n\t python csv2mat.py datafile.csv targetfile.csv dstfile.mat \n'
        sys.exit()
    src1, src2, dst = argv[1], argv[2], argv[3]
    combine_save_aggr_main(src1, src2, dst)



if __name__ == '__main__':
    main(sys.argv)
