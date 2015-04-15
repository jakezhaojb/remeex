# -*- coding: utf-8 -*-
# Jake

''' Main file '''

import os
import sys
import json
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


def combine_save_aggr_stride(data_file, target_file, folder_name):
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
    num = data.shape[0]
    for i in range(num):
        filename_mat = os.path.join(folder_name, str(i)+'.mat')
        filename_t7b = os.path.join(folder_name, str(i)+'.t7b')
        dict_save = {}
        # Aggregate
        data_elem = data[i, :]
        target_elem = target[i, :]
        dict_save['1'] = data_elem.reshape(data_elem.size, 1)
        dict_save['2'] = target_elem.reshape(target_elem.size, 1)
        save_to_mat(dict_save, filename_mat)
        os.system('th mat2t7b.lua --src ' + filename_mat + ' --dst ' + filename_t7b)
        os.system('rm ' + filename_mat)


def combine_save_aggr_main(data_folder, target_folder, folder_name, stride_flag=False):  # TODO this is ugly
    data_file_list = os.listdir(data_folder)
    data_file_list = data_file_list[::-1]
    target_file_list = os.listdir(target_folder)
    target_file_list = target_file_list[::-1]
    if os.path.exists(folder_name):
        print '%s already exists.' % folder_name
    def iter(args):
        data_file = args[0]
        target_file = args[1]
        assert data_file == target_file
        if data_file.find('.csv') != -1:
            data_file_nocsv = data_file[:data_file.find('.csv')]
        else:
            data_file_nocsv = data_file
        subfolder_name = os.path.join(folder_name, data_file_nocsv)
        data_file_complete = os.path.join(data_folder, data_file)
        target_file_complete = os.path.join(target_folder, target_file)
        if os.path.exists(subfolder_name):
            print '%s already done' % data_file
            return
        if stride_flag:
            combine_save_aggr_stride(data_file_complete, target_file_complete, subfolder_name)
        else:
            combine_save_aggr(data_file_complete, target_file_complete, subfolder_name)
        print "%s done." % data_file
    map(iter, zip(data_file_list, target_file_list))


def main(argv):
    # TODO this is ugly, rewrite it by ArgsParsers
    # TODO write a force flag
    if not len(argv) == 5:
        print 'Usage: \n\t python main.py /path/to/data/folder /path/to/target/folder /path/to/t7b/folder stride_flag \n'
        sys.exit()
    src1, src2, dst, flag = argv[1], argv[2], argv[3], argv[4]
    assert(flag in ['false', 'true'])
    flag = json.loads(flag)
    combine_save_aggr_main(src1, src2, dst, stride_flag=flag)


if __name__ == '__main__':
    main(sys.argv)
