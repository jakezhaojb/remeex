import librosa
import sys
sys.path.append('../data')
import load_cqt
sys.path.append('../')
import utils
import numpy as np
import os
import pickle
import gc

class args():
    def __init__(self,kX,dX,kY,dY,split):
        self.kX=kX
        self.dX=dX
        self.kY=kY
        self.dY=dY
        self.split=split

OPT={}
#OPT[1] = args(kX=5,  dX=1, kY=5,  dY=1, split='train')
OPT[2] = args(kX=11, dX=1, kY=84,  dY=1, split='train')
OPT[3] = args(kX=5,  dX=1, kY=84, dY=1, split='train')
OPT[4] = args(kX=11, dX=1, kY=5,  dY=1, split='train')
OPT[5] = args(kX=21, dX=1, kY=5,  dY=1, split='train')

cqt_folderpath = '/home/jmj418/cqt2'
datasplits_filepath = '../data/datasplits.txt'
out_dir = '/scratch/jmj418/data/gmm'


for opt in OPT.values():
    kX=opt.kX; dX=opt.dX; kY=opt.kY; dY=opt.dY; split=opt.split 

    os.system('mkdir -p %s' % out_dir)
    filename = 'obs_%dx%d_%dx%d' % (kX,dX,kY,dY)
    path = os.path.join(out_dir,filename+'.npy')
    pathmeta = os.path.join(out_dir,filename+'.pk1')

    if os.path.exists('%s' % path):
        answer = input("%s already exists.  Delete? (y/n)" % path)
        if answer == 'y':
            os.system('rm %s' % path)
        else:
            sys.exit('aborting')
    os.system('rm -f %s' % pathmeta)

    print '\n>> kX=%dx%d kY=%dx%d' % (kX,dX,kY,dY),'\n'

    CQT = load_cqt.load_cqt(cqt_folderpath,datasplits_filepath)
    obs = []

    if split == 'train':
        SONGS = CQT.splits.train
    elif split == 'validation':
        SONGS = CQT.splits.validation
    elif split == 'test':
        SONGS = CQT.splits.test
    else:
        sys.exit('unknown split type')

    for i,song in enumerate(SONGS):
        gc.collect()
        print ">> generating cqt arrays",i,"of",len(SONGS)
        cqt = CQT.load(song)
        n,p = cqt.shape

        iX = np.arange(0,n-kX+1,dX)
        iY = np.arange(0,p-kY+1,dY)

        for x in iX:
            for y in iY:
                obs.append(cqt[x:x+kX,y:y+kY].ravel())
    print '\n>>',len(obs),"obs"
    obs = np.vstack(obs)
    gc.collect()

    print '>> shape =',obs.shape

    print '>> saving to disk'
    meta = [obs.shape,obs.dtype,opt]
    utils.pickleIt(meta,pathmeta)
    obs.tofile(path)




