
import sys
import numpy as np 
import h5py, threading
import queue as Queue
import h5py, glob
from pathlib import Path
import random
import matplotlib.pylab as plb


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
    X, Y = None, None
    with h5py.File(data_file_h5, 'r') as hdf_fd:
        X = hdf_fd['images'][:].astype(np.float32)
        Y = hdf_fd['labels'][:].astype(np.float32)
        print("X shape: ", X.shape)
    while True:
        idx = np.random.randint(0, X.shape[0]-in_depth, mb_size)
        if(X.shape[1]-img_size <= 0):
            crop_idx = 0
        else:
            crop_idx = np.random.randint(0, X.shape[1]-img_size)
        batch_X = np.array([X[s_idx : (s_idx+in_depth)] for s_idx in idx])
        batch_X = batch_X[:, :, crop_idx:(crop_idx+img_size), crop_idx:(crop_idx+img_size)]
        c=0
        print(c)
        batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 1)
        batch_Y = batch_Y[:, :, crop_idx:(crop_idx+img_size), crop_idx:(crop_idx+img_size)]
        c+=1
        yield batch_X, batch_Y

if __name__ == '__main__':
    mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(data_file_h5="datasets/train_2d.h5", mb_size=2, in_depth=3, img_size=512), max_prefetch=16)  
    for i in range(2):
        x, y = mb_data_iter.next()
        print("X shape: ", x.shape)
    # print size of mb_data_iter
    # print(sys.getsizeof(mb_data_iter))


    # c=0
    # while(True):
    #     if X_mb is None:
    #         break
    #     c+=1
    # print("Total number of batches: ", c)
