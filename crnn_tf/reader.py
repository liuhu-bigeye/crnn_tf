import sys
import h5py
import json
import random
import os
import cv2
import glog
import pdb
import pickle
import numpy as np
import threading
from time import sleep

data_dir = '/mnt/disk1/liuhu/data/mnt/ramdisk/max/90kDICT32px'
lexicon_file = os.path.join(data_dir, 'lexicon.txt')
sys.path.insert(0, '/home/liuhu/workspace/ctc_ocr/crnn_tf')

with open(lexicon_file) as f:
    lexicon = [l.strip() for l in f.readlines()]
    max_length = max([len(l) for l in lexicon])


def char2token(char):
    if ord(char)<=ord('9'):
        return ord(char)-ord('0')+1
    else:
        return 10+ord(char)-ord('a')+1

def word2token(word):
    return [char2token(c) for c in word]

# from utils import sparse_tuple_from as sparse_tuple_from
from utils_multi import *

class Reader(threading.Thread):
    """
    This class is designed to automatically feed mini-batches.
    The reader constantly monitors the state of the variable 'data_buffer'.
    When finding the 'data_buffer' is None, the reader will fill a mini-batch into it.
    This is done in the backend, i.e. the reader is in an independent thread.
    For users, they only need to call iterate_batch() to get a new mini-batch.
    """
    daemon = True

    def __init__(self, phase, batch_size, do_shuffle=False):
        # Initialization of super class
        threading.Thread.__init__(self)
        annotation_file = os.path.join(data_dir, 'annotation_mini_%s.txt'%phase)

        # load database
        assert os.path.exists(annotation_file)
        # log_self(__file__)
        self.phase = phase
        self.do_shuffle = do_shuffle
        self.db = {'imglist':[], 'label':[]}

        with open(annotation_file) as f:
            for l in f.readlines():
                imgname, tag = l.strip().split(' ')
                label = lexicon[int(tag)]
                self.db['imglist'].append(imgname)
                self.db['label'].append(label)

        self.batch_size = batch_size
        self.img_size = (32, 100)
        self.vocabulary_size = 10 + 26
        self.n_samples = len(self.db['label'])

        # initialization
        self.running = True
        self.data_buffer = None
        self.lock = threading.Lock()

        self._shuffle()
        self.start()

    def _shuffle(self):
        """
        shuffle the data, called when one pass is over or at the beginning of training
        """
        index = range(self.n_samples)
        if self.do_shuffle:
            random.shuffle(index)

        self.n_batch = int(self.n_samples / self.batch_size)
        self.data_flow = [index[self.batch_size*i: self.batch_size*(i+1)] for i in range(self.n_batch)]
        self.i_batch = 0

    def run(self):
        """
        over write the 'run' method of threading.Thread
        """
        while self.running:
            if self.data_buffer is None:
                indices = self.data_flow[self.i_batch]
                token = sparse_tuple_from([np.array(word2token(self.db['label'][i]), dtype=np.int32) for i in indices])

                image = self.get_imgs(indices)
                mask = np.ones((self.batch_size, 26), dtype=np.float32)
                data = [image, mask, token]

                self.i_batch += 1
                if self.i_batch >= self.n_batch:
                    self._shuffle()

                with self.lock:
                    self.data_buffer = data
            sleep(0.0001)

    def get_imgs(self, indices):
        images = np.concatenate([cv2.resize(cv2.imread(os.path.join(data_dir, self.db['imglist'][i]), cv2.IMREAD_GRAYSCALE), self.img_size) for i in indices], axis=0)
        images = images.reshape((-1, self.img_size[1], self.img_size[0])).transpose([0, 2, 1])
        return images.astype(np.float32)

    def next_batch(self):
        while self.data_buffer is None:
            sleep(0.0001)
        d = self.data_buffer
        with self.lock:
            self.data_buffer = None
        return d

    def close(self):
        self.running = False
        self.join()

if __name__ == '__main__':
    train_set = Reader(phase='train', batch_size=2, do_shuffle=True, resample=True, distortion=True)
    for i in range(train_set.n_batch):
        inputs = train_set.next_batch()
        glog.info([s.shape for s in inputs[:-1]])

    exit(0)