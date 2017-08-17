#!/usr/bin/env python

import os
import pdb
import glog
import h5py
import numpy as np
np.set_printoptions(precision=4)

import sys
sys.path.insert(0, '/home/liuhu/workspace/fight_4_icml/')
from loader_config import Config
from evaluate.eval_utils import convert_pred_to_hypo, evaluate

def estimate(model, dataset, config, epoch):
    estimate_path = dataset.estimate_path
    estimate_key = dataset.estimate_key
    estimate_df = h5py.File(estimate_path, 'a')

    begin_index = 0
    do_shuffle = dataset.do_shuffle
    dataset.do_shuffle = False
    for index, inputs in enumerate(dataset.iterate(epoch=epoch/config.items['em_rate'])):
        estimate_preds, = model.estimate_func(*inputs)
        batch_size, h_len, t_len = estimate_preds.shape	# blank already at the first of axis -1

        end_index = begin_index + batch_size
        estimate_feats = np.ones((batch_size, dataset.max_h_len, dataset.max_t_len + 1))
        estimate_feats[:, :h_len, :t_len] = estimate_preds
        estimate_df[estimate_key][begin_index: end_index] = estimate_feats

        if end_index%400 == 0:
            glog.info('estimated %d outof %d samples...' % (end_index, dataset.n_samples))
        begin_index = end_index

    estimate_df.close()
    dataset.do_shuffle = do_shuffle

def prob_predict(model, dataset, config, epoch):
    predict_df = h5py.File(config.items['model']+'.prediction.h5', 'w')
    predict_key = dataset.phase
    if not predict_key in predict_df.keys():
        predict_df.create_dataset(predict_key, (dataset.n_samples, dataset.max_h_len, dataset.vocabulary_size+1))

    begin_index = 0
    do_shuffle = dataset.do_shuffle
    dataset.do_shuffle = False
    for index, inputs in enumerate(dataset.iterate(epoch=epoch/config.items['em_rate'])):
        token = inputs[2]
        prob_preds, best_path_token, ctc_loss, best_path_loss, greedy_loss = model.prob_func(*inputs)
        pdb.set_trace()

        batch_size, max_h_len, voc_size = prob_preds.shape	# blank already at the first of axis -1
        assert voc_size == dataset.vocabulary_size + 1

        end_index = begin_index + batch_size
        predict_df[predict_key][begin_index: end_index, :max_h_len] = prob_preds

        if end_index%400 == 0:
            glog.info('predict %d outof %d samples...' % (end_index, dataset.n_samples))
        begin_index = end_index

    predict_df.close()
    dataset.do_shuffle = do_shuffle

def valid(model, valid_set, config, epoch):
    hypos, IDs, losses = [], [], np.array([])
    # inputs: feat, mask, label, estimate, indices, ID
    # returns: pred_argmax
    for valid_inputs in valid_set.iterate(epoch=epoch / config.items['em_rate']):
        valid_return = model.predict_func(*valid_inputs)
        losses = np.vstack((losses, valid_return[:-1])) if losses.shape != (0,) else np.array(valid_return[:-1])

        h_len = np.sum(valid_inputs[1], axis=1)
        hypo = convert_pred_to_hypo(valid_return[-1], h_len)
        hypos.extend(hypo)
        IDs.extend(valid_inputs[-1])

    hypotheses = dict(zip(IDs, hypos))
    WER = evaluate(hypotheses, valid_set.vocabulary, os.path.join(config.items['snap_path'], 'output'), epoch, valid_set.phase)
    glog.info('%s set, Epoch %d, WER = %f, losses = %s'%(valid_set.phase, epoch, WER, np.mean(losses, axis=0)))
    return WER


def train_valid(model, train_set, valid_set, test_set, config):
    min_wer = [np.inf, np.inf]	# valid_wer, test_wer
    patienced_epoch = 0
    min_wer_epoch = -1
    phase = config.items['phase']
    glog.info('Phase starts with %s'%phase)

    for epoch in range(config.items['starting'], config.items['max_epoch']):
        # may change phase between ctc and em
        if epoch % config.items['em_rate'] == 0 and epoch > config.items['begin_loops']:
            config.load_config()
            if phase != config.items['phase']:
                phase = config.items['phase']
                glog.info('Change phase to %s'%phase)

            # estimate
            estimate(model, train_set, config, epoch)
            estimate(model, valid_set, config, epoch)
            estimate(model, test_set, config, epoch)

        # valid and test
        WER_valid = valid(model, valid_set, config, epoch)
        WER_test = valid(model, test_set, config, epoch)

        # dealing with patience
        if WER_valid < min_wer[0]:
            min_wer = [WER_valid, WER_test]
            min_wer_epoch = epoch
            patienced_epoch = 0
        else:
            patienced_epoch += 1
            if patienced_epoch > config.items['patience']:
                patienced_epoch = 0
                model.set_learning_rate(np.array(model.learning_rate.get_value() * 0.2, dtype=np.float32))
        glog.info('No WER down for %d epoch, patienced for %d epoch, min_wer: %s' % (epoch - min_wer_epoch, patienced_epoch, min_wer))

        losses = np.array([])
        if phase == 'origin_ctc':
            train_func = model.origin_ctc_func
        elif phase == 'em_ctc':
            train_func = model.maximize_func
        else:
            assert False

        for iter, train_inputs in enumerate(train_set.iterate(epoch=epoch/config.items['em_rate'])):
            train_return = train_func(*train_inputs)
            losses = np.vstack((losses, train_return)) if losses.shape!=(0,) else np.array(train_return)
            if iter % config.items['disp_iter']==0:
                glog.info('Phase = %s, Epoch %d/%d, Iteration %d, lr = %.2e, Training loss = %s' %
                          (phase, epoch/config.items['em_rate'], epoch, iter, model.learning_rate.get_value().astype(np.float32), np.mean(losses, axis=0)))
                model.set_learning_rate()
                losses = np.array([])

        # snapshot
        save_file = os.path.join(config.items['snap_path'], config.items['prefix']+'_epoch_%04d' % epoch)
        glog.info('Snap to path: %s' % save_file)
        model.save_model(save_file)
        model.set_learning_rate()
    glog.info('done, min WER = dev: %f, test: %f, at epoch %d' % (min_wer[0], min_wer[1], min_wer_epoch))

def main():
    if len(sys.argv) == 3:
        config = Config(sys.argv[1], sys.argv[2])
    else:
        assert False
    from utils import mkdir_safe, log_self
    log_self(__file__)

    glog.info('generating model...')
    from model_after import Model
    model = Model(learning_rate=config.items['lr'], config=config)

    # load model
    if 'model' in config.items.keys():
        glog.info('loading model: %s...' % config.items['model'])
        model.load_model(config.items['model'])
    elif 'model_old' in config.items.keys():
        glog.info('loading model from old: %s...' % config.items['model_old'])
        model.load_old_model(config.items['model_old'])

    from reader import Reader
    train_set = Reader(phase='train', config=config, do_shuffle=True, resample=True)
    valid_set = Reader(phase='dev', config=config, do_shuffle=True, resample=False, feature_mean=train_set.feature_mean, feature_std=train_set.feature_std)
    test_set = Reader(phase='test', config=config, do_shuffle=True, resample=False, feature_mean=train_set.feature_mean, feature_std=train_set.feature_std)

    try:
        config.items['starting'] = int(config.items['model'].split('_')[-1])
    except:
        config.items['starting'] = 0

    if 'predict' in config.items.keys():
        prob_predict(model, train_set, config, epoch=config.items['starting'])
        prob_predict(model, valid_set, config, epoch=config.items['starting'])
        prob_predict(model, test_set, config, epoch=config.items['starting'])
        return

    # snapshot path
    mkdir_safe(config.items['snap_path'])
    mkdir_safe(os.path.join(config.items['snap_path'], 'output_dev'))
    mkdir_safe(os.path.join(config.items['snap_path'], 'output_test'))

    glog.info('training...')
    train_valid(model, train_set, valid_set, test_set, config)

    glog.info('end')

if __name__ == '__main__':
    main()
