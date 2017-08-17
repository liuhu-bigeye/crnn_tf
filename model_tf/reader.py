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

data_dir = '/home/trunk/disk1/database-rwth-2014/phoenix2014-release/'
database_file = os.path.join(data_dir, 'database_2014_combine.pkl')
data_file = os.path.join(data_dir, 'feat_multimodal.h5')
sys.path.insert(0, '/home/liuhu/workspace/journal/all_vgg_s/model_tf')

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

    def __init__(self, phase, batch_size, c3d_depth=4, depth_stride=4, resample_at_end=False, resample=False, distortion=False, do_shuffle=False):
        # Initialization of super class
        threading.Thread.__init__(self)

        # load database
        assert os.path.exists(database_file)
        # log_self(__file__)
        self.phase = phase
        with open(database_file) as f:
            db = pickle.load(f)
            self.db = db[phase]
            # {'folder': [], 'signer': [], 'annotation': [], 'vocabulary': [], 'token': [], 'begin_index': [], 'end_index': []}

        self.tokens = self.db['token']
        self.vocabulary = self.db['vocabulary']

        h = h5py.File(data_file)

        self.image = [h['right_images'], h['left_images']]
        self.oflow = [h['right_of'], h['left_of']]
        self.coord = [h['right_coord'], h['left_coord'], h['head_coord']]

        self.batch_size = batch_size

        # PCA on RGB pixels for color shifts
        px = self.image[0][sorted(np.random.choice(799006, 300, replace=False))]
        px = np.concatenate((px, self.image[1][sorted(np.random.choice(799006, 300, replace=False))]), axis=0)
        px = px.reshape((-1, 3)) / 255.
        px -= px.mean(axis=0)
        self.eig_value, self.eig_vector = np.linalg.eig(np.cov(px.T))

        with open('/home/trunk/disk1/database-rwth-2014/phoenix2014-release/coord.pkl') as f:
            d = pickle.load(f)
            self.mean_coord = np.array(d['coord_mean'], dtype=np.float32)
            self.std_coord = np.array(d['coord_std'], dtype=np.float32)

        self.img_size = 224
        self.resample_at_end = resample_at_end
        self.resample = resample
        self.distortion = distortion
        self.do_shuffle = do_shuffle
        self.c3d_depth = c3d_depth
        self.depth_stride = depth_stride
        self.vocabulary_size = len(self.db['vocabulary'])

        self.n_samples = len(self.db['folder'])
        h_len = [int(np.ceil(float(e - b - self.c3d_depth) / float(self.depth_stride))) + 1 for b, e in zip(self.db['begin_index'], self.db['end_index'])]
        X_len = [(l - 1) * self.depth_stride + self.c3d_depth for l in h_len]

        self.max_h_len = max(h_len)
        self.max_X_len = max(X_len)

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
                y_len = np.array([len(self.db['token'][i]) for i in indices], dtype=np.int32)
                max_y_len = max(y_len)
                token = sparse_tuple_from([np.array(self.db['token'][i], dtype=np.int32)+1 for i in indices])

                # token = np.array([np.concatenate([self.tokens[idx], (max_y_len-y_len[i])*[-2]])+1 for i,idx in enumerate(indices)], dtype=np.int32)

                ID = [self.db['folder'][i] for i in indices]
                b_idx = [b for b in self.db['begin_index']]
                e_idx = self.db['end_index']

                x_len = [e_idx[i] - b_idx[i] for i in indices]

                if self.resample_at_end:    # for estimate
                    h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride))) + 1 for l in x_len]
                    X_Len = [(l - 1) * self.depth_stride + self.c3d_depth for l in h_len]
                    upsamp_indices = upsampling_at_end(x_len, self.c3d_depth, self.depth_stride)
                elif self.resample == True: # for training
                    X_Len, upsamp_indices = resampling(x_len, self.c3d_depth, self.depth_stride)
                    h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride)))+1 for l in X_Len]
                else:                       # for test and final prediction
                    h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride)))+1 for l in x_len]
                    X_Len = [(l-1)*self.depth_stride + self.c3d_depth for l in h_len]
                    upsamp_indices = upsampling(x_len, self.c3d_depth, self.depth_stride)

                # pdb.set_trace()
                max_h_len = np.max(h_len)
                max_X_Len = np.max(X_Len)

                image = np.zeros((self.batch_size, max_X_Len, self.img_size, self.img_size, 3), dtype=np.float32)
                # oflow = np.zeros((batch_size, max_X_Len, 2, 2, self.img_size, self.img_size), dtype=np.float32)
                # coord = np.zeros((batch_size, 20, max_X_Len), dtype=np.float32)
                mask = [np.concatenate((np.ones(l), np.zeros(max_h_len - l))) for l in h_len]

                # gathering features
                for i, ind in enumerate(indices):
                    image_raw, warp_mat = self.get_imgs(b_idx[ind], e_idx[ind]) # (x_len, 2, 3, self.img_size, self.img_size)
                    image_aug = interp_images(image_raw, upsamp_indices[i])  # (X_Len, 2, 3, self.img_size, self.img_size)  # interp_images means interp without float
                    assert X_Len[i] == image_aug.shape[0]
                    image[i, :X_Len[i]] = image_aug

                    # oflow_raw = self.get_oflow(b_idx[ind], e_idx[ind], warp_mat)
                    # oflow_aug = interp_images(oflow_raw, upsamp_indices[i])  # (X_Len, 2, 2, self.img_size, self.img_size)
                    # assert X_Len[i] == oflow_aug.shape[0]
                    # oflow[i, :X_Len[i]] = oflow_aug

                    # coord_raw = self.get_coord(range(b_idx[ind], e_idx[ind]))   # have problem due to position augmentation
                    # coord_aug = interp_images(coord_raw, upsamp_indices[i])  # (X_Len, 20)
                    # coord[i, :, :X_Len[i]] = coord_aug.transpose([1, 0])


                # oflow = np.transpose(oflow, (2, 0, 1, 3, 4, 5))  # (2, batch_size, max_X_Len, 2, self.img_size, self.img_size)

                # image = np.reshape(np.float32(image), (-1, 3, self.img_size, self.img_size))  # (2 * batch_size * X_len, 3, self.img_size, self.img_size)
                # oflow = np.reshape(np.float32(oflow), (-1, 2, self.img_size, self.img_size)) / 20. * 128.  # (2 * batch_size * X_len, 2, self.img_size, self.img_size)
                # coord = np.float32(coord)
                mask = np.array(mask, dtype=np.float32)  # (batch_size, max_h_len)

                data = [image, mask, token]

                self.i_batch += 1
                if self.i_batch >= self.n_batch:
                    self._shuffle()

                with self.lock:
                    self.data_buffer = data
            sleep(0.0001)

    def get_imgs(self, begin_index, end_index):
        # same augmentation for one sentence
        # return RGB images
        imgs = np.zeros((end_index-begin_index, self.img_size, self.img_size, 3), dtype=np.float32)
        imgs_orig = np.zeros((end_index-begin_index, 96, 96, 3), dtype=np.float32)

        mean_file = np.array([123, 117, 102], dtype=np.float32)
        for k in xrange(1):
            imgs_orig = self.image[k][begin_index: end_index].astype(np.float32)
            for i in range(end_index-begin_index):
                imgs[i] = cv2.resize(imgs_orig[i] - mean_file[None, None, :], (self.img_size, self.img_size))

        imgs = np.reshape(imgs, (-1, self.img_size, self.img_size, 3))
        imgs, mat = im_augmentation(imgs, self.eig_value, self.eig_vector, trans=0.1, color_dev=0.2, distortion=self.distortion)
        return imgs, mat

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