import sys
import collections
import logging
import pdb
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception

from keras.utils.conv_utils import convert_kernel

# import metrics
# import sequence_layers
import utils

# Return outputs
OutputEndpoints = collections.namedtuple('OutputEndpoints', [
    'ctc_loss', 'prediction'
])

# TODO(gorban): replace with tf.HParams when it is released.
ModelParams = collections.namedtuple('ModelParams', [
    'vocabulary_size'
])


# def get_softmax_loss_fn(label_smoothing):
#   """Returns sparse or dense loss function depending on the label_smoothing.

#     Args:
#       label_smoothing: weight for label smoothing

#     Returns:
#       a function which takes labels and predictions as arguments and returns
#       a softmax loss for the selected type of labels (sparse or dense).
#     """
#   if label_smoothing > 0:
#     def loss_fn(labels, logits):
#       return (tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
#   else:
#     def loss_fn(labels, logits):
#       return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#   return loss_fn


class Model(object):
  """Class to create the All GoogleNet LSTM Model."""

  def __init__(self,
               vocabulary_size=1295,
               mparams=None):

    super(Model, self).__init__()
    self._params = ModelParams(vocabulary_size=vocabulary_size)
    self.learning_rate = 1e-5

    self.images = tf.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 3], name='images')
    self.seqs_length = tf.placeholder(dtype=tf.int32, shape=[None], name='seqs_length')
    self.targets = tf.sparse_placeholder(tf.int32, name='targets')

    self._build_training()

    # # Saver
    # with tf.device('/cpu:0'):
    #   self.saver = tf.train.Saver(tf.all_variables())


  def _build_training(self):
    net = {}
    net['input'] = self.images

    input_shape = tf.shape(net['input'])
    self.batch_size, self.max_X_len = input_shape[0], input_shape[1]
    net['input_reshape'] = tf.reshape(self.images, [-1, 224, 224, 3])

    num_out = self.batch_size * self.max_X_len
    # Googlenet
    # with tf.variable_scope('conv_tower/INCE'):
    #   with slim.arg_scope(inception.inception_v3_arg_scope()):
    #     net['feat_mixed_7c_0'], _ = inception.inception_v3_base(net['input_reshape'][:num_out], final_endpoint='Mixed_7c')

    with tf.variable_scope('conv_tower/INCE'):
      net['inception5'], _ = inception.inception_v1_old_base(net['input_reshape'], final_endpoint='inception5')
      net['avgpool_0a_7x7'] = slim.avg_pool2d(net['inception5'], [7, 7], stride=1, scope='AvgPool_0a_7x7')

    # pdb.set_trace()
    # net['fc_mini'] = tf.contrib.layers.fully_connected(tf.reshape(net['feat_mixed_7c'], [self.batch_size * self.max_X_len, 5*5*1280]), num_outputs=1024, scope='fc_mini')
    net['feat_flattened'] = tf.reshape(net['avgpool_0a_7x7'], [self.batch_size, self.max_X_len, 1024])

    # pdb.set_trace()
    # Conv1d, Maxpool1d
    net['conv1d_1'] = tf.layers.conv1d(net['feat_flattened'], filters=1024, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv1d_1')
    net['pool1d_1'] = tf.layers.max_pooling1d(net['conv1d_1'], pool_size=2, strides=2)

    net['conv1d_2'] = tf.layers.conv1d(net['pool1d_1'], filters=1024, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv1d_2')
    net['pool1d_2'] = tf.layers.max_pooling1d(net['conv1d_2'], pool_size=2, strides=2)
    net['lstm_input_0'] = net['pool1d_2']
    # pdb.set_trace()

    # LSTM
    num_layers = 2
    for n in range(num_layers):
      # Define LSTM cell
      lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=512, state_is_tuple=True)
      lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, input_keep_prob=0.5, output_keep_prob=0.5)

      lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=512, state_is_tuple=True)
      lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, input_keep_prob=0.5, output_keep_prob=0.5)

      outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=net['lstm_input_%d'%n], sequence_length=self.seqs_length, scope='BLSTM_%d'%n, dtype=tf.float32, time_major=False)

      output = tf.concat(values=outputs, axis=2)
      if n < num_layers - 1:
        net['lstm_input_%d'%(n+1)] = output
      else:
        net['lstm_output'] = tf.reshape(output, [-1, 512*2])

    # pdb.set_trace()
    # Output
    net['fc_output'] = tf.contrib.layers.fully_connected(net['lstm_output'], num_outputs=self._params.vocabulary_size+1, activation_fn=None, scope='fc_output')
    net['output'] = tf.reshape(net['fc_output'], [self.batch_size, -1, self._params.vocabulary_size+1])

    self.net = net
    logits = tf.transpose(net['output'], [1, 0, 2])
    loss = tf.nn.ctc_loss(self.targets, logits, self.seqs_length)
    cost = tf.reduce_mean(loss, name='cost')

    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.train_step = self.optimizer.minimize(cost)

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, self.seqs_length)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.targets), name='ler')

  def train_minibatch(self, session, images, seqs_length, targets):
      # Inputs needed in the computation
      feed_dict = {self.images: images,
                   self.seqs_length: seqs_length,
                   self.targets: targets}

      # Operations to be executed
      fetches = ['cost:0', 'ler:0', self.train_step]
      # Run the session
      result = session.run(fetches, feed_dict)
      # Return training loss
      return result[:2]

  def assign_from_pkl(self, pkl_path):
    with open(pkl_path, 'rb') as f:
      load_variables = pickle.load(f)

    uninitialized_vars = []
    for i, variable in enumerate(tf.global_variables()):
      # 0 -41
      # 42-77 + 10
      # 78-117+ 20
      if i<=41:
        idx = i
      elif i<=77:
        idx = i + 10
      elif i<=117:
        idx = i + 20
      else:
        uninitialized_vars.append(variable)
        continue
      
      variable_shape = load_variables[idx].shape
      if len(variable_shape) == 1:
        load_variable = load_variables[idx]
      elif len(variable_shape) == 4:
        load_variable = np.transpose(load_variables[idx], [3, 2, 1, 0])
      elif len(variable_shape) == 3:
        load_variable = np.transpose(load_variables[idx], [2, 1, 0])
      else:
        assert False

      print variable.name, variable.get_shape(), load_variable.shape
      variable.assign(load_variable).op.run()

    pdb.set_trace()
    tf.initialize_variables(uninitialized_vars).op.run()
    return

  def set_learning_rate(self, to_lr=None):
    pass

if __name__ == '__main__':
  # sess_config = tf.ConfigProto(device_count = {'GPU': 0})
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    with tf.device('/cpu:0'):
      model = Model()
      model.assign_from_pkl('/home/liuhu/workspace/journal/model_tf/cnn_all_vgg_s_epoch_0004_iter_010000')


