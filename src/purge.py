#-*- coding: utf-8 -*-
# Jake

'''
This file will purge the model savings, and only keep the latest trained 3 models
'''

import os
import re
import sys


def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))


def purge(path):
    files = sorted_ls(path)
    files = filter(lambda x: re.match('epoch_\d{1,2}0.t7b', x), files)
    files = files[:-3]  # Important!! save 3 latest models.
    for file_elem in files:
        abs_file_elem = os.path.join(path, file_elem)
        os.system('rm ' + abs_file_elem)


def navigate(path):
    subfolder = os.listdir(path)
    for subfolder_elem in subfolder:
        abs_subfolder_elem = os.path.join(path, subfolder_elem)
        if not os.path.isdir(abs_subfolder_elem):
            continue
        purge(abs_subfolder_elem)
    print 'Done.'


def main(argv):
    assert len(argv) == 2
    assert argv[1].startswith('/scratch/jakez/') or argv[1].startswith('/scratch/jz1672/') 
    navigate(argv[1])


if __name__ == '__main__':
    main(sys.argv)

