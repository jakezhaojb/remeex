from time import time
from data_loader import *
from data_proc import *
from dpark import DparkContext


def load_anno_test():
    all_generator = read_mdb_all_data_generator()
    start = time()
    anno_aggr = read_mdb_melody_anno_aggr(all_generator)
    end = time()
    print "Time spent on reading all melody: %i" % (end - start)

def load_anno_mfcc_test():
    all_generator = read_mdb_all_data_generator()
    anno_aggr = read_mdb_mix_audio_aggr(all_generator)
    dpark_ctx = DparkContext()
    start = time()
    y = feature_mfcc_aggr(dpark_ctx, anno_aggr)
    end = time()
    print "Time spent on reading all melody: %i" % (end - start)

def main():
    load_anno_mfcc_test()

if __name__ == '__main__':
    main()
