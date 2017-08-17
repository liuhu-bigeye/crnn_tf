import os
import glog
import theano
import theano.tensor as T
import numpy as np
from skimage.transform import resize

def process_image(images, image_mean, shape):
	images = images[:, :, :, [2, 1, 0]]  # BGR to RGB, (X_Len, 128, 128, 3)

	if (not shape == images.shape[1]) or (not shape == images.shape[2]):
		images_reshape = np.zeros((images.shape[0], shape, shape, 3), dtype=np.float32)
		for n in range(images.shape[0]):
			im = resize(images[n] / 255.0, (shape, shape)) * 255.0
			images_reshape[n] = im

		images = images_reshape
	images = images - image_mean  # (X_Len, shape, shape, 3)

	return images


def resampling(lengths, depth, stride, temp_scaling=0.2):
	indices = []
	samp_lengths = []
	min_len = 32
	max_len = 300

	L = 1.0 - temp_scaling
	U = 1.0 + temp_scaling

	for length in lengths:
		new_length = int(length * (L + (U - L) * np.random.random()))

		if new_length < min_len:
			new_length = min_len
		if new_length > max_len:
			new_length = max_len

		if (new_length - depth) % stride != 0:
			new_length += stride - (new_length - depth) % stride
		samp_lengths.append(new_length)

		if new_length <= length:
			index = sorted(np.random.choice(length, new_length, replace=False))
		else:
			index = list((np.sort(np.random.random(new_length) * (length - 1)) + np.linspace(0, length-1, new_length)) / 2.0)

		indices.append(index)

	return samp_lengths, indices


def resampling_fixed(lengths, depth, stride, temp_scaling=0.2, seeds = None):
	indices = []
	samp_lengths = []
	min_len = 32
	max_len = 300

	L = 1.0 - temp_scaling
	U = 1.0 + temp_scaling

	for i, length in enumerate(lengths):
		np.random.seed(seeds[i])
		new_length = int(length * (L + (U - L) * np.random.random()))

		if new_length < min_len:
			new_length = min_len
		if new_length > max_len:
			new_length = max_len

		if (new_length - depth) % stride != 0:
			new_length += stride - (new_length - depth) % stride
		samp_lengths.append(new_length)

		if new_length <= length:
			index = sorted(np.random.choice(length, new_length, replace=False))
		else:
			index = list((np.sort(np.random.random(new_length) * (length - 1)) + np.linspace(0, length-1, new_length)) / 2.0)

		indices.append(index)

	return samp_lengths, indices

def upsampling(lengths, depth, stride):
	indices = []
	for length in lengths:
		if length <= depth/2:
			k = np.random.randint(0, depth - length)
			index = sorted(range(length) + [0] * k + [length-1] * (depth-length-k))
			indices.append(index)
			continue
		elif length < depth:
			add_ind = np.random.choice(length-1, depth-length, replace=False)
		elif (length - depth) % stride != 0:
			add_ind = np.random.choice(length-1, stride-(length-depth)%stride)
		else:
			add_ind = []

		index = sorted(range(length) + [k+0.5 for k in add_ind])
		indices.append(index)

	return indices

def upsampling_fixed(lengths, depth, stride, seeds = None):
	indices = []
	for i, length in enumerate(lengths):
		np.random.seed(seeds[i])
		if length <= depth/2:
			k = np.random.randint(0, depth - length)
			index = sorted(range(length) + [0] * k + [length-1] * (depth-length-k))
			indices.append(index)
			continue
		elif length < depth:
			add_ind = np.random.choice(length-1, depth-length, replace=False)
		elif (length - depth) % stride != 0:
			add_ind = np.random.choice(length - 1, stride - (length - depth) % stride)
		else:
			add_ind = []

		index = sorted(range(length) + [k+0.5 for k in add_ind])
		indices.append(index)

	return indices

def gaussian_noise_fixed(feat, scale, seeds):
	if scale == 0.0:
		return
	assert feat.shape[0] == len(seeds)
	for i in range(feat.shape[0]):
		np.random.seed(seeds[i])
		feat[i] += np.random.normal(scale=scale, size=feat[i].shape)

def interp_locations(arr, index):
	# arr: (len, 6)
	ndim = arr.shape[1]
	arr_interp = np.zeros((len(index), ndim), dtype=np.float32)

	for i, ind in enumerate(index):
		if ind == int(ind):
			arr_interp[i] = arr[ind]
		else:
			arr_interp[i] = arr[int(ind)] * (np.ceil(ind) - float(ind)) + arr[int(ind)+1] * (float(ind) - np.floor(ind))

	return arr_interp


def interp_images(arr, index):
	# arr: (len, 128, 128, 3)
	arr_interp = np.float32([arr[int(round(ind))] for ind in index])

	return arr_interp


def diff_locations(arr):
	# arr: (len, d)
	arr_diff = np.diff(arr, axis=0)
	output = np.zeros_like(arr)
	len, dim = arr.shape
	for i in range(dim):
		output[:, i] = np.interp(range(len), range(1, len), arr_diff[:, i])

	return output


def calc_location_mean_val(db, feat):
	n_samples = len(db['folder'])
	data_agg = np.array([])
	b_idx = db['begin_index']
	e_idx = db['end_index']
	for i in range(n_samples):
		loc_raw = feat['coords'][b_idx[i]: e_idx[i]]  # (len, 6)

		data = np.zeros((loc_raw.shape[0], 20), dtype=np.float32)
		data[:, 0: 2] = loc_raw[:, 0: 2]
		data[:, 2: 4] = loc_raw[:, 2: 4]
		data[:, 4: 6] = loc_raw[:, 0: 2] - loc_raw[:, 4: 6]
		data[:, 6: 8] = loc_raw[:, 2: 4] - loc_raw[:, 4: 6]
		data[:, 8:10] = loc_raw[:, 0: 2] - loc_raw[:, 2: 4]

		data[:, 10: ] = diff_locations(data[:, : 10])
		data_agg = np.vstack((data_agg, data)) if data_agg.size else data

	data_mean = np.mean(data_agg, axis=0)
	data_std = np.std(data_agg, axis=0)

	return data_mean, data_std


def mkdir_safe(path):
	if not os.path.exists(path):
		os.makedirs(path)


def Softmax(h):
	import theano.tensor as T
	dimlist = list(T.xrange(h.ndim))
	dimlist[-1] = 'x'
	h_normed = h - T.max(h, axis=-1).dimshuffle(dimlist)
	out = T.exp(h_normed) / T.exp(h_normed).sum(axis=-1).dimshuffle(dimlist)
	return out

def softmax_np(h):
	h = h - np.max(h, axis=-1)[:, :, None]
	out = np.exp(h) / np.exp(h).sum(axis=-1)[:, :, None]
	return out

def log_safe(h, eps = 1e-35):
	eps_T = theano.shared(np.float32(eps))
	h_log = T.log(eps_T + h)
	return h_log

def divide_safe(a, b, eps = 1e-30):
	eps_T = theano.shared(np.float32(eps))
	answer = (a)/(b+eps_T)
	return answer

def log_self(file):
	filename = os.path.abspath(file)
	if filename.endswith('c'):
		filename = filename[:-1]
	with open(filename) as f:
		glog.info(f.read())


class Rander(object):
	def __init__(self, epoch, n_samples):
		seed = epoch
		a = 3
		c = 5
		m = 982451653

		start_n_samples = 40
		self.n_samples = n_samples
		self.random_seeds = np.empty(n_samples + start_n_samples, dtype=np.int64)
		for i in range(n_samples + start_n_samples):
			seed = (a*seed + c)%m
			self.random_seeds[i] = seed

		self.random_seeds = self.random_seeds[start_n_samples:]

	def get(self, indexs):
		indexs = np.array(indexs, dtype=np.int32)
		assert indexs.min()>=0 and indexs.max()<self.n_samples
		return self.random_seeds[indexs]

if __name__ == '__main__':
	rander = Rander(1, 10)
	for i in range(10):
		print rander.get(i)
