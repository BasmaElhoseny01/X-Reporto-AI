
import sys
import numpy as np 
import h5py, threading
import queue as Queue
import h5py, glob
from pathlib import Path
import random
import matplotlib.pylab as plb
from src.denoiser.utils import *


class bkgdGen(threading.Thread):
    def __init__(self, data_generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = data_generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            # block if necessary until a free slot is available
            self.queue.put(item, block=True, timeout=None)
        self.queue.put(None)

    def next(self):
        # block if necessary until an item is available
        next_item = self.queue.get(block=True, timeout=None)
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

def gen_train_batch_bg(data_file_h5, mb_size, in_depth, img_size):
    #TODO:
    # just one image as inout and one image as output dont make this rubbish
    X, Y = None, None
    # with h5py.File(data_file_h5, 'r') as hdf_fd:
    with h5py.File("datasets/noisy4test.h5", 'r') as hdf_fd:
        X = hdf_fd['images'][:].astype(np.float32)
        # Y = hdf_fd['labels'][:].astype(np.float32)
    with h5py.File("datasets/clean4test.h5", 'r') as hdf_fd:
        # Y = hdf_fd['labels'][:].astype(np.float32)
        Y = hdf_fd['images'][:].astype(np.float32)
        
    while True:
        idx = np.random.randint(0, X.shape[0]-in_depth, mb_size)
        print("idx: ", idx)
        if(X.shape[1]-img_size <= 0):
            crop_idx = 0
        else:
            crop_idx = np.random.randint(0, X.shape[1]-img_size)
        batch_X = np.array([X[s_idx : (s_idx+in_depth)] for s_idx in idx])
        # batch_X = batch_X[:, :, crop_idx:(crop_idx+img_size), crop_idx:(crop_idx+img_size)]
        batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 1)
        # batch_Y = batch_Y[:, :, crop_idx:(crop_idx+img_size), crop_idx:(crop_idx+img_size)]
        yield batch_X, batch_Y

def get1batch4test(data_file_h5, in_depth):
    # X = h5py.File(x_fn, 'r')['images']
    # Y = h5py.File(y_fn, 'r')['images']
    with h5py.File(data_file_h5, 'r') as hdf_fd:
        X = hdf_fd['images'][:].astype(np.float32)
        Y = hdf_fd['labels'][:].astype(np.float32)

    idx = (X.shape[0]//2, )
    batch_X = np.array([X[s_idx : (s_idx+in_depth)] for s_idx in idx])
    batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 1) 
    return batch_X.astype(np.float32) , batch_Y.astype(np.float32)


def get_predictions_batch(x_fn, in_depth):
    X = None
    with h5py.File(x_fn, 'r') as hdf_fd:
        X = hdf_fd['images'][:].astype(np.float32)
    
    L = X.shape[0]
    size = X.shape[1]
    for i in range(X.shape[0]):
        if i - in_depth//2 >=0 and i + in_depth//2 < L and in_depth %2 == 0:
            batch_X = X[i-in_depth//2: i+in_depth//2]
        elif i - in_depth//2 >=0 and i + in_depth//2 < L and in_depth %2 == 1:
            batch_X = X[i-in_depth//2: i+in_depth//2+1]
        elif i + in_depth//2 >= L and in_depth %2 == 0:
            batch_X = np.concatenate((X[i-in_depth//2:L], X[:i+in_depth//2-L]), axis=0)
        elif i + in_depth//2 >= L and in_depth %2 == 1:
            batch_X = np.concatenate((X[i-in_depth//2:L], X[:i+in_depth//2-L+1]), axis=0)
        else:
            if in_depth %2 == 0:
                batch_X = np.concatenate((X[L+i-in_depth//2:L], X[0:i+in_depth//2]), axis=0)
            else:
                batch_X = np.concatenate((X[L+i-in_depth//2:L], X[0:i+in_depth//2+1]), axis=0)
        batch_X = np.transpose(batch_X, (1, 2, 0))
        yield np.expand_dims(batch_X, 0), np.zeros((1, size, size, 1))

if __name__ == '__main__':
    mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(data_file_h5="datasets/train_noise.h5", mb_size=2, in_depth=3, img_size=512), max_prefetch=16)  
    i=0
    x_prev_0=0
    x_curr_200=0
    x, y = mb_data_iter.next()
    print("x shape: ", x.shape)
    save2image(y[0,0,:,:], "y.png")
    save2image(x[0,0,:,:], "x_1.png")
    save2image(x[0,1,:,:], "x_2.png")
    save2image(x[0,2,:,:], "x_3.png")

    # while True:
    #     print(i)
    #     if i==0:
    #         x_prev_0=y
    #     if i==200:
            # x_curr_200=y
        #     if np.mean(np.abs(x_prev_0 - x_curr_200)) ==0:
        #         print("zero at ",i)
        #     sys.exit()
        # i+=1
        # print("X shape: ", x.shape)

