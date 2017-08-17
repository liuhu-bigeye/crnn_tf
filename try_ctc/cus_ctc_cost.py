import numpy as np
import pdb
import glog
import theano
import theano.tensor as T
# from theano_ctc import ctc_cost

from utils import Softmax, softmax_np, log_safe, divide_safe
floatX = theano.config.floatX
intX = np.int32

import sys
sys.path.insert(0, '/home/liuhu/tools/rnn_ctc/nnet')

def em_ctc_cost(e_pred, pred, pred_len, token, blank):
	'''
	ctc_cost of only one sentence
	:param e_pred: (T, nb, voca_size + 1)
	:param pred: (T, nb, voca_size + 1)					(4,1,3)
	:param pred_len: (nb,)	pred_len of prediction		(1)
	:param token: (nb, U)	-1 for NIL					(1,2)
	:param blank: (1)
	:return: ctc_cost
	'''
	nb, U = token.shape[0], token.shape[1]
	token_len = T.sum(T.neq(token, -1), axis=-1)

	# token_with_blank
	token = token[:, :, None]	# (nb, U, 1)
	token_with_blank = T.concatenate((T.ones_like(token, dtype='int32')*blank, token), axis=2).reshape((nb, 2*U))
	token_with_blank = T.concatenate((token_with_blank, T.ones((nb, 1), dtype='int32')*blank), axis=1)	# (nb, 2*U+1)
	length = token_with_blank.shape[1]

	# only use these predictions
	pred = pred[:, T.arange(nb, dtype='int32')[:, None], token_with_blank]	# (T, nb, 2U+1)
	e_pred = e_pred[:, T.arange(nb, dtype='int32')[:, None], token_with_blank]	# (T, nb, 2U+1)

	# recurrence relation
	sec_diag = T.concatenate((T.zeros((nb, 2), dtype=intX), T.neq(token_with_blank[:, :-2], token_with_blank[:, 2:])), axis=1) * T.neq(token_with_blank, blank)	# (nb, 2U+1)
	recurrence_relation = T.tile((T.eye(length) + T.eye(length, k=1)), (nb, 1, 1)) + T.tile(T.eye(length, k=2), (nb, 1, 1))*sec_diag[:, None, :]	# (nb, 2U+1, 2U+1)
	recurrence_relation = recurrence_relation.astype(floatX)

	# alpha for estimate
	alpha = T.zeros_like(token_with_blank, dtype=floatX)
	alpha = T.set_subtensor(alpha[:, :2], e_pred[0, :, :2])################(nb, 2U+1)	p
	# beta for maximize
	beta = T.zeros_like(token_with_blank, dtype=floatX)
	beta = T.set_subtensor(beta[:, :2], e_pred[0, :, :2]*log_safe(pred[0, :, :2]))################(nb, 2U+1) e_p * log(p)

	# dynamic programming
	# (T, nb, 2U+1)
	(probability_alpha, probability_beta), _ = theano.scan(compute_one_step, sequences=[e_pred[1:], pred[1:]], outputs_info=[alpha, beta], non_sequences=[recurrence_relation])

	# estimate prob
	labels_e_2 = probability_alpha[pred_len - 2, T.arange(nb, dtype='int32'), 2 * token_len - 1]
	labels_e_1 = probability_alpha[pred_len - 2, T.arange(nb, dtype='int32'), 2 * token_len]
	labels_e_prob = labels_e_2 + labels_e_1

	# maximize prob
	labels_m_2 = probability_beta[pred_len - 2, T.arange(nb, dtype='int32'), 2 * token_len - 1]
	labels_m_1 = probability_beta[pred_len - 2, T.arange(nb, dtype='int32'), 2 * token_len]
	labels_m_prob = labels_m_2 + labels_m_1

	cost = -divide_safe(labels_m_prob, labels_e_prob)
	return cost

def compute_one_step(e_prob, prob, alpha, beta, recurrence_relation):
	alpha = T.batched_dot(alpha, recurrence_relation) * e_prob
	beta = T.batched_dot(beta, recurrence_relation) * e_prob + log_safe(prob) * alpha
	return alpha, beta

def generate_data(T, nb, length_max, voca_size):
	# generate preds(nb, T, voca_size), tokens(no 0, -1 for null), lengths(length for preds)

	# preds = softmax_np(np.random.random((T, nb, voca_size+1)))			# (T, nb, voca_size + 1)
	# e_preds = softmax_np(np.random.random((T, nb, voca_size+1)))			# (T, nb, voca_size + 1)
	#
	# length = np.random.randint(2, length_max+1, size=(nb))				# (nb)
	# tokens = np.array([np.concatenate([np.random.randint(voca_size, size=l), -np.ones(length_max-l)]) for l in length])
	#
	# return e_preds.astype(floatX), preds.astype(floatX), tokens.astype(intX), length.astype(intX)

	e_preds = np.array([[0.1,0.1,0.8],[0.2,0.1,0.7],[0.3,0.4,0.3],[0.5,0.2,0.3]], dtype=floatX)
	preds = np.array([[0.2,0.5,0.3],[0.7,0.2,0.1],[0.1,0.3,0.6],[0.2,0.3,0.5]], dtype=floatX)
	tokens = np.array([1,2], dtype=intX)
	length = np.array(4, dtype=intX)
	return e_preds[:, None, :], preds[:, None, :], tokens[None, :], length[None]

if __name__ == '__main__':
	e_pred = T.tensor3('e_pred')
	pred = T.tensor3('pred')
	token = T.imatrix('token')
	length = T.ivector('length')

	cost = em_ctc_cost(e_pred, pred, length, token, blank=0)

	f_m_ctc = theano.function(inputs=[e_pred, pred, token, length], outputs=[cost])

	nb = 50
	T = 50
	length_max = 20
	voca_size = 1000

	for i in range(1):
		e_preds, preds, tokens, length = generate_data(T, nb, length_max, voca_size)
		glog.info('start testing...')
		glog.info('%s'%(f_m_ctc(e_preds, preds, tokens, length)))
