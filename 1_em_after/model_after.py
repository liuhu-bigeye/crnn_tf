import os
import sys
import pdb
import glog
import pickle
import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
sys.path.insert(0, '/home/liuhu/workspace/fight_4_icml')

# import caffe

from loader_config import Config
from try_ctc.cus_ctc_cost import em_ctc_cost
from try_ctc.m_ctc_cost import ctc_cost, best_right_path_cost, greedy_cost
from utils import *

class Model(object):
    def __init__(self, learning_rate, config, vocabulary_size=1295):
        # need to be same voca_size and hidde_ndim so as to load same shape params
        log_self(__file__)
        # model paras
        self.config = config

        self.alpha = np.array(1e-3, dtype=np.float32)
        self.eps = np.array(1e-6, dtype=np.float32)
        self.learning_rate = theano.shared(np.float32(learning_rate))
        if type(config.items['hidden_ndim'])!=list:
            self.hidden_ndim = [config.items['hidden_ndim']]
        else:
            self.hidden_ndim = config.items['hidden_ndim']
        if config.items['highway'] == True:
            prepare_lstm_dim = config.items['hidden_ndim'][0]*2
        else:
            prepare_lstm_dim = 1024

        self.nClasses = vocabulary_size + 1
        self.vocabulary_size = vocabulary_size

        # variables
        feat = T.tensor3('fc6')    # (nb, max_Xlen, 1024)
        estimate = T.tensor3('estimate')    # (nb, T, voca_size + 1)
        mask = T.matrix('mask')  # (nb, max_hlen)
        token = T.imatrix('token')
        self.nb = mask.shape[0]
        self.max_hlen = mask.shape[1]

        net = {}
        # encoding network for features
        net['data'] = InputLayer(shape=(None, None, 1024))
        net['conv1d_rh'] = Conv1DLayer(DimshuffleLayer(net['data'], (0, 2, 1)), num_filters=1024, filter_size=3, pad='same')    # (nb, 1024, len)
        net['pool1d_rh'] = MaxPool1DLayer(net['conv1d_rh'], pool_size=2)
        # net['drop1d_rh'] = DropoutLayer(net['pool1d_rh'], p=0.1)

        net['conv1d_2'] = Conv1DLayer(net['pool1d_rh'], num_filters=prepare_lstm_dim, filter_size=3, pad='same')
        net['pool1d_2'] = MaxPool1DLayer(net['conv1d_2'], pool_size=2)    #(nb, 1024, max_hlen)
        # net['drop1d_2'] = DropoutLayer(net['pool1d_2'], p=0.1)

        # LSTM
        net['lstm_prepare_0'] = DimshuffleLayer(net['pool1d_2'], (0, 2, 1))
        net['mask'] = InputLayer(shape=(None, None))  # (nb, max_hlen)

        for idx, hidden_ndim in enumerate(self.hidden_ndim):
            net['lstm_frw_%d'%idx] = LSTMLayer(incoming=net['lstm_prepare_%d'%idx], mask_input=net['mask'], forgetgate=Gate(b=lasagne.init.Constant(1.0)), num_units=hidden_ndim)  # (nb, max_hlen,hidden_ndim)
            net['lstm_bck_%d'%idx] = LSTMLayer(incoming=net['lstm_prepare_%d'%idx], mask_input=net['mask'], forgetgate=Gate(b=lasagne.init.Constant(1.0)), num_units=hidden_ndim, backwards=True)
            net['lstm_prepare_%d'%(idx+1)] = ConcatLayer((net['lstm_frw_%d'%idx], net['lstm_bck_%d'%idx]), axis=2)
            if config.items['highway'] == True:
                net['lstm_prepare_%d'%(idx+1)] = ElemwiseSumLayer([net['lstm_prepare_%d'%idx], net['lstm_prepare_%d'%(idx+1)]])

        net['lstm_shp'] = ReshapeLayer(net['lstm_prepare_%d'%len(self.hidden_ndim)], shape=(-1, 2*self.hidden_ndim[-1]))  # (nb*max_hlen, 2*self.hidden_ndim[-1])
        net['out'] = DenseLayer(net['lstm_shp'], self.nClasses, nonlinearity=identity)  # (nb*max_hlen, nClasses)
        net['out_lin'] = ReshapeLayer(net['out'], shape=(self.nb, -1, self.nClasses))

        self.net = net
        self.name_layers()
        self.try_sl_model()

        output_lin = get_output(self.net['out_lin'], {self.net['data']: feat, self.net['mask']: mask})    # no dropout
        output_softmax = Softmax(output_lin)
        # (nb, max_hlen, voca_size+1)

        # EM CTC
        # estimate
        token_and_zero = T.concatenate([T.zeros((self.nb, 1), dtype='int32'), token], axis=1)    # (nb, max_t+1)
        e_output = output_softmax[T.arange(self.nb, dtype='int32')[:, None, None],
                                       T.arange(self.max_hlen)[None, :, None],
                                       token_and_zero[:, None, :]]  # (nb, max_hlen, max_t+1)

        # maximize
        pred = output_softmax.transpose([1, 0, 2])    # (nb, voca_size + 1, max_hlen)
        e_pred = estimate.transpose([1, 0, 2])**self.config.items['em_hard_rate']        # (nb, voca_size + 1, max_hlen)
        em_ctc_loss = em_ctc_cost(e_pred, pred, mask.sum(axis=1).astype('int32'), token, blank=0)


        # Origin CTC
        ctc_loss = ctc_cost(pred, mask.sum(axis=1).astype('int32'), token, blank=0)

        self.params = get_all_params(net['out_lin'], trainable=True)
        regular_loss = lasagne.regularization.apply_penalty(get_all_params(net['out_lin'], regularizable=True), lasagne.regularization.l2) \
                  * np.array(5e-4 / 2, dtype=np.float32)

        em_ctc_loss_all_mean = em_ctc_loss.mean() + regular_loss
        ctc_loss_all_mean = ctc_loss.mean() + regular_loss


        # Best right path cost
        best_right_path_loss, best_path_token = best_right_path_cost(pred, mask, token, blank=0)
        greedy_loss = greedy_cost(pred, mask)

        # Input and Output
        self.inputs_estimate = [feat, mask, token]
        self.outputs_estimate = [e_output]

        self.inputs_prob = [feat, mask, token]
        self.outputs_prob = [output_softmax, best_path_token, ctc_loss, best_right_path_loss, greedy_loss]

        self.inputs_maximize = [feat, mask, token, estimate]
        self.outputs_maximize = [em_ctc_loss_all_mean, em_ctc_loss.mean(), ctc_loss_all_mean, ctc_loss.mean(),
                                 best_right_path_loss.mean(), greedy_loss.mean()]
        self.updates_maximize = lasagne.updates.adam(em_ctc_loss_all_mean, self.params, learning_rate=self.learning_rate)

        self.input_origin_ctc = [feat, mask, token, estimate]
        self.output_origin_ctc = [em_ctc_loss_all_mean, em_ctc_loss.mean(), ctc_loss_all_mean, ctc_loss.mean(),
                                  best_right_path_loss.mean(), greedy_loss.mean()]
        self.updates_origin_ctc = lasagne.updates.adam(ctc_loss_all_mean, self.params, learning_rate=self.learning_rate)

        self.inputs_predict = [feat, mask, token, estimate]
        self.outputs_predict = [em_ctc_loss_all_mean, em_ctc_loss.mean(), ctc_loss_all_mean, ctc_loss.mean(),
                                best_right_path_loss.mean(), greedy_loss.mean(), output_lin.argmax(axis=-1)]

        glog.info('Model built')

    def try_sl_model(self):
        # return
        # try save load model
        dummy_save_file = os.path.join(self.config.output_path,'dummy.pkl')
        glog.info('try save load dummy model to: %s...' % dummy_save_file)
        self.save_model(dummy_save_file)
        self.load_model(dummy_save_file)
        # os.system('rm -rf %s'%dummy_save_file)
        # glog.info('dummy save load success, remove it and start calculate outputs...')

    # input always be feat, mask, label, estimate, indices, ID
    def estimate_func(self, *inputs):
        if not hasattr(self, 'estimate_function'):
            glog.info('making estimate function...')
            self.estimate_function = theano.function(inputs=self.inputs_estimate, outputs=self.outputs_estimate)
        return self.estimate_function(*inputs[:3])

    def prob_func(self, *inputs):
        if not hasattr(self, 'prob_function'):
            glog.info('making prob function...')
            self.prob_function = theano.function(inputs=self.inputs_prob, outputs=self.outputs_prob)
        return self.prob_function(*inputs[:3])

    def maximize_func(self, *inputs):
        if not hasattr(self, 'maximize_function'):
            glog.info('making maximize function...')
            self.maximize_function = theano.function(inputs=self.inputs_maximize, outputs=self.outputs_maximize, updates=self.updates_maximize)
        return self.maximize_function(*inputs[:4])

    def origin_ctc_func(self, *inputs):
        if not hasattr(self, 'origin_ctc_function'):
            glog.info('making origin ctc function...')
            self.origin_ctc_function = theano.function(inputs=self.input_origin_ctc, outputs=self.output_origin_ctc, updates=self.updates_origin_ctc)
        return self.origin_ctc_function(*inputs[:4])

    def predict_func(self, *inputs):
        if not hasattr(self, 'predict_function'):
            glog.info('making predict function...')
            self.predict_function = theano.function(inputs=self.inputs_predict, outputs=self.outputs_predict)
        return self.predict_function(*inputs[:4])

    def name_layers(self):
        for k in self.net.keys():
            self.net[k].name = k
            for i in range(len(self.net[k].params.keys())):
                self.net[k].params.keys()[i].name = '%s_%s'%(k, self.net[k].params.keys()[i].name)
                glog.info('set network layer name: %s'%self.net[k].params.keys()[i].name)

    def load_model(self, model_file):
        with open(model_file) as f:
            params_dict = pickle.load(f)
        params_full = get_all_params(self.net['out_lin'], trainable=True)
        for param in params_full:
            param.set_value(params_dict[param.name])
            glog.info('load param %s'%param.name)
        glog.info('load model from %s' % os.path.basename(model_file))

    def load_old_model(self, model_file):
        with open(model_file) as f:
            params_0 = pickle.load(f)
            params_1 = pickle.load(f)
        lasagne.layers.set_all_param_values(self.net['out_lin'], params_0[-4:]+params_1)
        glog.info('load model from %s' % os.path.basename(model_file))

    def save_model(self, model_file):
        params_full = get_all_params(self.net['out_lin'], trainable=True)
        params_dict = {p.name:p.get_value() for p in params_full}
        with open(model_file, 'wb') as f:
            pickle.dump(params_dict, f)

    def set_learning_rate(self, to_lr=None):
        if not to_lr is None:
            self.learning_rate.set_value(to_lr)
            glog.info('Auto change learning rate to %.2e' % to_lr)
        else:
            self.config.load_config()
            if 'lr_change' in self.config.items.keys():
                lr = np.float32(self.config.items['lr_change'])
                if not lr == self.learning_rate:
                    self.learning_rate.set_value(lr)
                    glog.info('Change learning rate to %.2e' % lr)
