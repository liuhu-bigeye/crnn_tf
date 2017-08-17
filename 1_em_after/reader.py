import os
import cv2
import pdb
import sys
import copy
import h5py
# import glog
import pickle
# import numpy as np
from loader_config import Config

from scipy.stats import norm
sys.path.insert(0, '/home/liuhu/workspace/fight_4_icml')
from utils import *

data_dir = '/home/trunk/disk1/database-rwth-2014/phoenix2014-release'
database_file = os.path.join(data_dir, 'database_2014_combine.pkl')


class Reader(object):
    def __init__(self, phase, config, c3d_depth=4, depth_stride=4, do_shuffle=False, resample=False, feature_mean=None, feature_std=None):
        # load database
        assert os.path.exists(database_file)
        log_self(__file__)

        self.eps = np.array(1e-35).astype(np.float32)
        self.phase = phase
        self.features = h5py.File(config.items['feature_path'])[config.items['feature_key']]

        # {'folder': [], 'signer': [], 'annotation': [], 'vocabulary': [], 'token': [], 'begin_index': [], 'end_index': []}
        with open(database_file) as f:
            db = pickle.load(f)
            self.db = db[phase]

        self.tokens = self.db['token']
        self.vocabulary = self.db['vocabulary']
        self.batch_size = config.items['batch_size']
        self.do_shuffle = do_shuffle
        self.c3d_depth = c3d_depth
        self.depth_stride = depth_stride
        self.vocabulary_size = len(self.db['vocabulary'])
        self.n_samples = len(self.db['folder'])

        x_len = [ei - bi for ei, bi in zip(self.db['end_index'], self.db['begin_index'])]
        h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride))) + 1 for l in x_len]
        self.max_h_len = int(max(h_len)*1.2) + 1
        self.max_t_len = max([len(t) for t in self.db['token']])
        glog.info('max hlen = %d, max tlen = %d'%(self.max_h_len, self.max_t_len))

        self.estimations = None
        self.estimate_path = config.items['estimate_path']
        self.estimate_key = 'esti_%s'%self.phase
        self.resample = resample
        self.gaussian_noise_scale = config.items['gaussian_noise_scale']

        if feature_mean is None:
            self.feature_mean = np.mean(self.features, axis=0).astype(np.float32)
            self.feature_std = np.std(self.features, axis=0).astype(np.float32)
        else:
            self.feature_mean = feature_mean
            self.feature_std = feature_std
        self.try_make_estimation()
        if 'begin_loops' in config.items:
            glog.info('%s set, initiate estimations...' % self.phase)
            self.initiate_estimations()

    def try_make_estimation(self):
        # only save estimations in token
        self.estimations = h5py.File(self.estimate_path, 'a')
        estimation_shape = (self.n_samples, self.max_h_len, self.max_t_len + 1)  # only need to save predictions in tokens

        if self.estimate_key in self.estimations.keys():
            glog.info('field %s already created, filling it...' % self.estimate_key)
        else:
            glog.info('Creating dataset to %s, key = %s, shape = %s' % (self.estimate_path, self.estimate_key, estimation_shape))
            self.estimations.create_dataset(self.estimate_key, estimation_shape)

            # close and reopen
            self.estimations.close()
            self.estimations = h5py.File(self.estimate_path, 'a')

    def iterate(self, epoch=0):
        rander = Rander(epoch, self.n_samples)

        index = range(self.n_samples)
        if self.do_shuffle:
            np.random.shuffle(index)

        for k in range(0, self.n_samples, self.batch_size):
            indices = index[k: k+self.batch_size]
            batch_size = len(indices)

            ID = [self.db['folder'][i] for i in indices]

            b_idx = self.db['begin_index']
            e_idx = self.db['end_index']

            x_len = [e_idx[i] - b_idx[i] for i in indices]
            if self.resample:
                X_Len, upsamp_indices = resampling_fixed(x_len, self.c3d_depth, self.depth_stride, seeds=rander.get(indices))
                h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride))) + 1 for l in X_Len]
            else:
                h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride))) + 1 for l in x_len]
                X_Len = [(l - 1) * self.depth_stride + self.c3d_depth for l in h_len]
                upsamp_indices = upsampling_fixed(x_len, self.c3d_depth, self.depth_stride, seeds=rander.get(indices))

            max_h_len = np.max(h_len)
            max_X_Len = np.max(X_Len)

            feat = np.zeros((batch_size, max_X_Len, 1024), dtype=np.float32)
            estimate = np.zeros((batch_size, max_h_len, self.vocabulary_size+1), dtype=np.float32)
            mask = [np.concatenate((np.ones(l), np.zeros(max_h_len - l))) for l in h_len]
            y_len = np.array([len(self.db['token'][i]) for i in indices], dtype=np.int32)
            label = -np.ones((batch_size, np.max(y_len)), dtype=np.int32)  # 0 for blank, -1 for no label

            # gathering features
            for i, ind in enumerate(indices):
                feat_raw = self.get_features(b_idx[ind], e_idx[ind])  # (x_len, 1024)
                feat_aug = interp_images(feat_raw, upsamp_indices[i])  # (X_Len, 1024)

                assert X_Len[i] == feat_aug.shape[0]
                feat[i] = np.concatenate((feat_aug, np.zeros((max_X_Len-X_Len[i], 1024), dtype=np.float32)), axis=0)  # (max_X_Len, 1024)

                token = self.tokens[ind]
                t_len = len(token)
                label[i, :t_len] = np.array(token, dtype=np.int32) + 1

                estimate[i, :, 0] = self.estimations[self.estimate_key][ind, :max_h_len, 0]
                estimate[i, np.arange(max_h_len)[:, None], label[i, :t_len][None, :]] = self.estimations[self.estimate_key][ind, :max_h_len, 1:t_len+1]

            gaussian_noise_fixed(feat, self.gaussian_noise_scale, seeds=rander.get(indices))
            mask = np.array(mask, dtype=np.float32)  # (batch_size, max_h_len)
            yield feat, mask, label, estimate, indices, ID
            # if phase == 'estimate':
            # 	yields = [feat, mask, label]
            # elif phase == 'maximize':
            # 	# pdb.set_trace()
            # 	# if epoch == 0:
            # 	# 	estimate = np.ones_like(estimate, dtype=np.float32)
            # 	yields = [feat, mask, estimate, label]
            # elif phase == 'predict':
            # 	yields = [feat, mask]
            # else:
            # 	assert False
            #
            # if return_indices:
            # 	yields += [np.array(indices, dtype=np.int32)]
            # if return_folder:
            # 	yields += [ID]
            #
            # yield yields

    def get_features(self, begin_index, end_index):
        feats = self.features[begin_index: end_index]
        return (feats - self.feature_mean) / (self.eps + self.feature_std)

    def initiate_estimations(self):
        self.estimations.close()
        self.estimations = h5py.File(self.estimate_path, 'a')

        scale = 0.5
        eps = 1e-3
        # estimation shape (n_sample, max_h_len, max_t_len + 1)
        for idx in range(self.n_samples):
            token = self.tokens[idx]
            t_len = len(token)
            h_len = int(np.ceil(float(self.db['end_index'][idx] - self.db['begin_index'][idx] - self.c3d_depth) / float(self.depth_stride))) + 1

            steps = np.linspace(0, h_len-1, t_len)
            esti = [np.ones(h_len)[:, None]*eps] + [norm.pdf(np.arange(h_len)[:, None], loc=round(l), scale=scale) for l in steps]
            esti = np.hstack(esti)

            self.estimations[self.estimate_key][idx, :h_len, :t_len + 1] = copy.deepcopy(esti / esti.sum(axis=-1)[:, None] * 0.9)
        self.estimations.close()
        self.estimations = h5py.File(self.estimate_path, 'a')

if __name__ == '__main__':
    config = Config('/home/liuhu/workspace/fight_4_icml/1_em_after/output/test_output', '/home/liuhu/workspace/fight_4_icml/configs/1_after/ctc_vs_em/config_initiate_em_rate4')
    train_set = Reader(phase='train', config=config, do_shuffle=True, resample=True)
