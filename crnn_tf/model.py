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

    self.images = tf.placeholder(dtype=tf.float32, shape=[None, 32, 100], name='images')
    self.seqs_length = tf.placeholder(dtype=tf.int32, shape=[None], name='seqs_length')
    self.targets = tf.sparse_placeholder(tf.int32, name='targets')

    self.ks = [3, 3, 3, 3, 3, 3, 2]
    self.ps = [1, 1, 1, 1, 1, 1, 0]
    self.ss = [1, 1, 1, 1, 1, 1, 1]
    self.nm = [64, 128, 256, 256, 512, 512, 512]
    self.nh = [256, 256]

    self._build_training()

    # # Saver
    # with tf.device('/cpu:0'):
    #   self.saver = tf.train.Saver(tf.all_variables())

  def convRelu(self, i, is_training=False, batch_norm=False):
    i -= 1
    self.net = tf.layers.conv2d(self.net, self.nm[i], [self.ks[i], self.ks[i]], padding=self.ps[i], stride=self.ss[i], activation_fn=False, scope='Conv2d_%d_%dx%d'%(i, self.ks[i], self.ks[i]))
    if batch_norm:
      self.net = tf.contrib.layers.batch_norm(self.net, center=True, scale=True, is_training=is_training, scope='BN_%d'%i)
    self.net = tf.nn.relu(self.net, 'relu')


  def blstm(self):
    num_layers = len(self.nh)
    # LSTM
    for i in range(num_layers):
      # Define LSTM cell
      lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=self.nh[i], state_is_tuple=True)
      lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, input_keep_prob=0.5, output_keep_prob=0.5)

      lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=self.nh[i], state_is_tuple=True)
      lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, input_keep_prob=0.5, output_keep_prob=0.5)

      outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=self.net, sequence_length=self.seqs_length, scope='BLSTM_%d' % i,
                                                   dtype=tf.float32, time_major=False)

      self.net = tf.concat(values=outputs, axis=2)
    self.net = tf.reshape(self.net, [-1, self.nh[-1] * 2])


  def _build_training(self, is_training):

    # input
    input_shape = tf.shape(self.images)
    self.batch_size, self.width, self.height = input_shape
    self.net = tf.reshape(self.images, [self.batch_size, self.width, self.height, 1])

    # rescale
    self.net = (self.net - 128.)/128.

    # conv1-7
    self.convRelu(1)
    self.net = tf.layers.max_pooling2d(self.net, pool_size=[2, 2], strides=[2, 2], scope='pool1')    # b*16*50*64

    self.convRelu(2)
    self.net = tf.layers.max_pooling2d(self.net, pool_size=[2, 2], strides=[2, 2], scope='pool2')    # b*8*25*128

    self.convRelu(3, is_training=is_training, batch_norm=True)
    self.convRelu(4)                                                                                 # b*8*25*256
    self.net = tf.pad(self.net, paddings=[[0, 0, 1, 0], [0, 0, 1, 0]])                               # b*8*27*256
    self.net = tf.layers.max_pooling2d(self.net, pool_size=[2, 2], strides=[2, 1], padding='VALID', scope='pool3')    # b*4*26*256

    self.convRelu(5, is_training=is_training, batch_norm=True)
    self.convRelu(6)
    self.net = tf.pad(self.net, paddings=[[0, 0, 0, 0], [0, 0, 1, 0]])                               # b*4*28*512
    self.net = tf.layers.max_pooling2d(self.net, pool_size=[2, 2], strides=[2, 1], padding='VALID', scope='pool4')    # b*2*27*512

    self.convRelu(7)                                                                                 # b*1*26*512
    self.net = tf.squeeze(self.net, [1])

    # BLSTM
    self.blstm()
    self.net = tf.contrib.layers.fully_connected(self.net, num_outputs=self._params.vocabulary_size+1, activation_fn=None, scope='fc_output')
    self.net = tf.reshape(self.net, [self.batch_size, -1, self._params.vocabulary_size+1])

    logits = tf.transpose(self.net, [1, 0, 2])
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


