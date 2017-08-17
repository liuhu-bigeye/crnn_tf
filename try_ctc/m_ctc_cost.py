import numpy as np
np.set_printoptions(precision=4)

import pdb
import glog
import theano
import theano.tensor as T
# from theano_ctc import ctc_cost

from utils import Softmax, softmax_np
floatX = theano.config.floatX
intX = np.int32

import sys
sys.path.insert(0, '/home/liuhu/tools/rnn_ctc/nnet')

def ctc_cost(pred, pred_len, token, blank):
    '''
    ctc_cost of multi sentences
    :param pred: (T, nb, voca_size+1)                    (4,1,3)
    :param pred_len: (nb,)    pred_len of prediction        (1)
    :param token: (nb, U)    -1 for NIL                    (1,2)
    :param blank: (1)
    :return: ctc_cost
    '''

    eps = theano.shared(np.float32(1e-35))
    nb, U = token.shape[0], token.shape[1]
    token_len = T.sum(T.neq(token, -1), axis=-1)

    # token_with_blank
    token = token[:, :, None]    # (nb, U, 1)
    token_with_blank = T.concatenate((T.ones_like(token, dtype=intX)*blank, token), axis=2).reshape((nb, 2*U))
    token_with_blank = T.concatenate((token_with_blank, T.ones((nb, 1), dtype=intX)*blank), axis=1)    # (nb, 2*U+1)
    length = token_with_blank.shape[1]

    # only use these predictions
    pred = pred[:, T.tile(T.arange(nb), (length, 1)).T, token_with_blank]    # (T, nb, 2U+1)

    # recurrence relation
    sec_diag = T.concatenate((T.zeros((nb, 2), dtype=intX), T.neq(token_with_blank[:, :-2], token_with_blank[:, 2:])), axis=1) * T.neq(token_with_blank, blank)    # (nb, 2U+1)
    recurrence_relation = T.tile((T.eye(length) + T.eye(length, k=1)), (nb, 1, 1)) + T.tile(T.eye(length, k=2), (nb, 1, 1))*sec_diag[:, None, :]    # (nb, 2U+1, 2U+1)
    recurrence_relation = recurrence_relation.astype(floatX)

    # alpha
    alpha = T.zeros_like(token_with_blank, dtype=floatX)
    alpha = T.set_subtensor(alpha[:, :2], pred[0, :, :2])################(nb, 2U+1)

    # dynamic programming
    # (T, nb, 2U+1)
    probability, _ = theano.scan(lambda curr, accum: T.batched_dot(accum, recurrence_relation) * curr, sequences=[pred[1:]], outputs_info=[alpha])

    labels_2 = probability[pred_len-2, T.arange(nb), 2*token_len-1]
    labels_1 = probability[pred_len-2, T.arange(nb), 2*token_len]
    labels_prob = labels_2 + labels_1


    cost = -T.log(labels_prob+eps)
    return cost

def best_right_path_cost(pred, mask, token, blank):
    '''
    best right path cost of multi sentences
    :param pred: (T, nb, voca_size+1)                    (4,1,3)
    :param mask: (nb, T)
    # :param pred_len: (nb,)    pred_len of prediction        (1)
    :param token: (nb, U)    -1 for NIL                    (1,2)
    :param blank: (1)

    :return: best_right_path_cost (nb,)
    :return: argmin_token (nb, T) best path, -1 for null
    '''

    pred_len = mask.sum(axis=-1).astype('int32')
    eps = theano.shared(np.float32(1e-35))
    EPS = theano.shared(np.float32(35))

    t = pred.shape[0]
    nb, U = token.shape[0], token.shape[1]
    token_len = T.sum(T.neq(token, -1), axis=-1)

    # token_with_blank
    token = token[:, :, None]    # (nb, U, 1)
    token_with_blank = T.concatenate((T.ones_like(token, dtype=intX)*blank, token), axis=2).reshape((nb, 2*U))
    token_with_blank = T.concatenate((token_with_blank, T.ones((nb, 1), dtype=intX)*blank), axis=1)    # (nb, 2*U+1)
    length = token_with_blank.shape[1]

    # only use these predictions
    pred = pred[:, T.tile(T.arange(nb), (length, 1)).T, token_with_blank]    # (T, nb, 2U+1)
    pred = -T.log(pred + eps)

    # recurrence relation
    sec_diag = T.concatenate((T.zeros((nb, 2), dtype=intX), T.neq(token_with_blank[:, :-2], token_with_blank[:, 2:])), axis=1) * T.neq(token_with_blank, blank)    # (nb, 2U+1)
    recurrence_relation = T.tile((T.eye(length) + T.eye(length, k=1)), (nb, 1, 1)) + T.tile(T.eye(length, k=2), (nb, 1, 1))*sec_diag[:, None, :]    # (nb, 2U+1, 2U+1)
    recurrence_relation = -T.log(recurrence_relation + eps).astype(floatX)

    # alpha
    alpha = T.ones_like(token_with_blank, dtype=floatX) * EPS
    alpha = T.set_subtensor(alpha[:, :2], pred[0, :, :2])################(nb, 2U+1)

    # dynamic programming
    # (T, nb, 2U+1)
    [log_probability, argmin_pos_1], _ = theano.scan(lambda curr, accum: ((accum[:, :, None] + recurrence_relation).min(axis=1) + curr, (accum[:, :, None] + recurrence_relation).argmin(axis=1)),
                                                   sequences=[pred[1:]], outputs_info=[alpha, None])

    # why pred_len-2?
    labels_1 = log_probability[pred_len-2, T.arange(nb), 2*token_len-1]		# (nb,)
    labels_2 = log_probability[pred_len-2, T.arange(nb), 2*token_len]		# (nb,)
    concat_labels = T.concatenate([labels_1[:, None], labels_2[:, None]], axis=-1)
    argmin_labels = concat_labels.argmin(axis=-1)

    cost = concat_labels.min(axis=-1)

    min_path = T.ones((t-1, nb), dtype=intX)*-1  # -1 for null
    min_path = T.set_subtensor(min_path[pred_len-2, T.arange(nb)], 2*token_len-1+argmin_labels)

    # (T-1, nb)
    min_full_path, _ = theano.scan(lambda m_path, argm_pos, m_full_path: argm_pos[T.arange(nb), T.maximum(m_path, m_full_path).astype('int32')].astype('int32'),
                                   sequences=[min_path[::-1], argmin_pos_1[::-1]], outputs_info=[min_path[-1]])
    argmin_pos = T.concatenate((min_full_path[::-1], min_path[-1][None, :]), axis=0) # (T, nb)
    argmin_token = token_with_blank[T.arange(nb)[None, :], argmin_pos]

    return cost, (argmin_token.transpose((1, 0))*mask + mask - 1).astype('int32')# alpha, log_probability, argmin_pos_1, argmin_labels, min_path, min_full_path, argmin_pos, token_with_blank, argmin_token

def greedy_cost(pred, mask):
    '''
    greedy loss of multiple sentences
    :param pred: (T, nb, voca_size+1)
    :param mask: (nb, T)
    :return: greedy_cost (nb,)
    '''
    greedy_pred = pred.max(axis=-1)
    greedy_pred = T.maximum(greedy_pred, 1 - mask.dimshuffle((1, 0)))
    log_greedy_pred = -T.log(greedy_pred)

    greedy_cost = log_greedy_pred.sum(axis=0)
    return greedy_cost

def generate_data(T, nb, length_max, voca_size):
    # generate preds(nb, T, voca_size), tokens(no 0, -1 for null), lengths(length for preds)

    preds = softmax_np(np.random.random((T, nb, voca_size+1)))            # (T, nb, voca_size + 1)
    assert length_max<=T and length_max>2
    length = np.random.randint(2, length_max+1, size=(nb))                # (nb)
    tokens = np.array([np.concatenate([np.random.randint(voca_size, size=l), -np.ones(length_max-l)]) for l in length])

    pred_len = np.zeros((nb))
    mask = np.zeros((nb, T))
    for i in range(nb):
        pred_len[i] = np.random.randint(length[i], T+1)
        mask[i, :pred_len[i]] = 1
    return preds.astype(floatX), tokens.astype(intX), pred_len.astype(intX), mask.astype(bool)

if __name__ == '__main__':
    pred = T.tensor3('pred')
    length = T.ivector('length')
    token = T.imatrix('token')

    cost, alpha, log_probability, argmin_pos_1, argmin_labels, min_path, min_full_path, argmin_pos, token_with_blank, argmin_token = best_right_path_cost(pred, length, token, blank=0)
    f_best_loss = theano.function([pred, length, token],
                                  [cost, alpha, log_probability, argmin_pos_1, argmin_labels, min_path, min_full_path, argmin_pos, token_with_blank, argmin_token])

    # (T, nb, voca_size+1)
    pred_np = np.array([[0.5, 0.4, 0.1],[0.3,0.1,0.6],[0.7,0.2,0.1],[0.3,0.5,0.2]]).astype(floatX)[:,None,:]
    # (nb)
    length_np = np.array([4]).astype(intX)
    # (nb, U)
    token_np = np.array([2,1]).astype(intX)[None,:]

    glog.info('%s, %s, %s'%(pred_np.shape, length_np.shape, token_np.shape))
    glog.info('%s'%zip('cost, alpha, log_probability, argmin_pos_1, argmin_labels, min_path, min_full_path, argmin_pos, token_with_blank, argmin_token'.split(', '),
                       f_best_loss(pred_np, length_np, token_np)))
    glog.info('cost_gt: %f'%( -np.log(0.105)))


    # preds = np.array([[0.1,0.1,0.8],[0.2,0.1,0.7],[0.3,0.4,0.3],[0.5,0.2,0.3]], dtype=floatX)
    # tokens = np.array([1,2], dtype=intX)
    # length = np.array(4, dtype=intX)
    # return preds[:, None, :], tokens[None, :], length[None]

# if __name__ == '__main__':
#     pred = T.tensor3('pred')
#     token = T.imatrix('token')
#     mask = T.imatrix('mask')
#     length = T.ivector('length')

#     # ctc_loss = ctc_cost(T.log(pred), token, length)
#     cost = ctc_cost(pred, length, token, blank=0)
#     best_loss, pred_, recurrence_relation, alpha, log_probability = best_right_path_cost(pred, length, token, blank=0)
#     greedy_loss = greedy_cost(pred, mask)

#     f_best_loss = theano.function(inputs=[pred, token, length], outputs=[best_loss, pred_, recurrence_relation, alpha, log_probability])
#     f_m_ctc = theano.function(inputs=[pred, token, length], outputs=[cost])
#     f_greedy = theano.function(inputs=[pred, mask], outputs=[greedy_loss])

#     nb = 5
#     T = 30
#     length_max = 10
#     voca_size = 50

#     for i in range(5):
#         preds, tokens, length, masks = generate_data(T, nb, length_max, voca_size)
#         glog.info('start testing...')
#         glog.info('%s'%(f_m_ctc(preds, tokens, length)))
#         glog.info('%s'%(f_best_loss(preds, tokens, length)[0]))
#         glog.info('%s'%(f_greedy(preds, masks)))
#         cost, pred, recurrence_relation, alpha, log_probability = f_best_loss(preds, tokens, length)
#         pdb.set_trace()
#         print cost.shape, pred.shape, recurrence_relation.shape, alpha.shape, log_probability.shape







