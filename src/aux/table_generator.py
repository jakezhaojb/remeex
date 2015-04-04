# -*- coding: utf-8 -*-
# Jake

import os
import sys


def get_file_list(folder, savepath, table_name):
    song_file_list = os.listdir(folder)
    if not savepath.endswith('.lua'):
        savepath += '.lua'
    file = open(savepath, 'w')
    file.write(table_name + '_idx_table = { \n')
    count = 0
    for song in song_file_list:
        # For each song
        if not os.path.isdir(os.path.join(folder, song)) or song.startswith('.'):
            continue
        
        num_file = len(os.listdir(os.path.join(folder, song)))
        file.write('\t [' + song + '] = ' + str(count))
        if song == song_file_list[-1]:
            file.write('}')
        else:
            file.write(', \n')
        count += num_file
    file.close()


def main(argv):
    if len(argv) != 4:
        print 'Usage: \n \t python table_generator /path/to/folder /path/to/save /name/of/table'
        return 
    get_file_list(argv[1], argv[2], argv[3])


if __name__ == '__main__':
    main(sys.argv)
