import os
import glog
import random
import numpy as np
import cv2
from skimage.transform import resize

# def process_image(images, image_mean, shape):
#     images = images[:, :, :, [2, 1, 0]]  # BGR to RGB, (X_Len, 128, 128, 3)
#
#     if (not shape == images.shape[1]) or (not shape == images.shape[2]):
#         images_reshape = np.zeros((images.shape[0], shape, shape, 3), dtype=np.float32)
#         for n in range(images.shape[0]):
#             im = resize(images[n] / 255.0, (shape, shape)) * 255.0
#             images_reshape[n] = im
#
#         images = images_reshape
#     images = images - image_mean  # (X_Len, shape, shape, 3)
#
#     return images

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def resampling(lengths, depth, stride, temp_scaling=0.2, max_len = 320):
    indices = []
    samp_lengths = []
    min_len = 32
    max_len = 120

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
            index = sorted(random.sample(range(length), new_length))
        else:
            index = list((np.sort(np.random.random(new_length) * (length - 1)) + np.linspace(0, length-1, new_length)) / 2.0)

        indices.append(index)

    return samp_lengths, indices


def upsampling(lengths, depth, stride):
    indices = []
    for length in lengths:
        if length <= depth/2:
            k = random.randint(0, depth - length)
            index = sorted(range(length) + [0] * k + [length-1] * (depth-length-k))
            indices.append(index)
            continue
        elif length < depth:
            add_ind = random.sample(range(length-1), depth - length)
        elif (length - depth) % stride != 0:
            add_ind = random.sample(range(length-1), stride - (length-depth) % stride)
        else:
            add_ind = []

        index = sorted(range(length) + [k+0.5 for k in add_ind])
        indices.append(index)

    return indices

def upsampling_at_end(lengths, depth, stride):
    indices = []
    for length in lengths:
        if length < depth:
            add_ind = [length - 1] * (depth - length)
        elif (length - depth) % stride != 0:
            add_ind = [length - 1] * (stride - (length-depth) % stride)
        else:
            add_ind = []

        index = sorted(range(length) + add_ind)
        indices.append(index)

    return indices


# def interp_locations(arr, index):
#     # arr: (len, 6)
#     ndim = arr.shape[1]
#     arr_interp = np.zeros((len(index), ndim), dtype=np.float32)
#
#     for i, ind in enumerate(index):
#         if ind == int(ind):
#             arr_interp[i] = arr[ind]
#         else:
#             arr_interp[i] = arr[int(ind)] * (np.ceil(ind) - float(ind)) + arr[int(ind)+1] * (float(ind) - np.floor(ind))
#
#     return arr_interp


def interp_images(arr, index):
    # arr: (len, 128, 128, 3)
    arr_interp = np.float32([arr[int(round(ind))] for ind in index])

    return arr_interp

def im_augmentation(ims_src, weight, vec, trans=0.1, color_dev=0.1, distortion=True):
    num, W, H, _ = ims_src.shape
    if distortion:
        ran_noise = np.random.random((4, 2))
        ran_color = np.random.randn(3,)
    else:
        ran_noise = np.ones((4, 2)) * 0.5
        ran_color = np.zeros(3,)

    # perspective translation
    dst = np.float32([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]) * np.float32([W, H])
    noise = trans * ran_noise * np.float32([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]]) * [W, H]
    src = np.float32(dst + noise)

    mat = cv2.getPerspectiveTransform(src, dst)
    for i in range(num):
        ims_src[i] = cv2.warpPerspective(ims_src[i], mat, (W, H))

    # color deviation
    deviation = np.dot(vec, (color_dev * ran_color * weight)) * 255.
    ims_src += deviation[None, None, None, :]

    return ims_src, mat


def of_augmentation(ims_src, mat):
    num, W, H, _ = ims_src.shape
    for i in xrange(num):
        ims_src[i] = cv2.warpPerspective(ims_src[i], mat, (W, H))

    return ims_src


def diff_locations(arr):
    # arr: (len, d)
    arr_diff = np.diff(arr, axis=0)
    output = np.zeros_like(arr)
    len, dim = arr.shape
    for i in range(dim):
        output[:, i] = np.interp(range(len), range(1, len), arr_diff[:, i])

    return output


# def calc_location_mean_val(db, feat):
#     n_samples = len(db['folder'])
#     data_agg = np.array([])
#     b_idx = db['begin_index']
#     e_idx = db['end_index']
#     for i in range(n_samples):
#         loc_raw = feat['coords'][b_idx[i]: e_idx[i]]  # (len, 6)
#
#         data = np.zeros((loc_raw.shape[0], 20), dtype=np.float32)
#         data[:, 0: 2] = loc_raw[:, 0: 2]
#         data[:, 2: 4] = loc_raw[:, 2: 4]
#         data[:, 4: 6] = loc_raw[:, 0: 2] - loc_raw[:, 4: 6]
#         data[:, 6: 8] = loc_raw[:, 2: 4] - loc_raw[:, 4: 6]
#         data[:, 8:10] = loc_raw[:, 0: 2] - loc_raw[:, 2: 4]
#
#         data[:, 10: ] = diff_locations(data[:, : 10])
#         data_agg = np.vstack((data_agg, data)) if data_agg.size else data
#
#     data_mean = np.mean(data_agg, axis=0)
#     data_std = np.std(data_agg, axis=0)
#
#     return data_mean, data_std

def mkdir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)


def Softmax(h):
    import theano.tensor as T
    dimlist = list(T.xrange(h.ndim))
    dimlist[-1] = 'x'
    h_normed = h - T.max(h, axis=-1).dimshuffle(dimlist)
    out = T.exp(h_normed) / T.exp(h_normed).sum(axis=-1).dimshuffle(dimlist)
    return out

def softmax_np(h):
    h_normed = h - np.expand_dims(h.max(axis=-1), -1)
    out = np.exp(h_normed) / np.expand_dims(np.exp(h_normed).sum(axis=-1), -1)
    return out

def log_self(file):
    filename = os.path.abspath(file)
    if filename.endswith('c'):
        filename = filename[:-1]
    with open(filename) as f:
        glog.info(f.read())


def set_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY+YG+GC+CB+BM+MR
    col = 0
    colorwheel = np.zeros((ncols, 3))

    colorwheel[col:col+RY, 0] = 255
    colorwheel[col:col+RY, 1] = [int(255*i/RY) for i in range(RY)]
    col = col+RY

    colorwheel[col:col+YG, 0] = [255-int(255*i/YG) for i in range(YG)]
    colorwheel[col:col+YG, 1] = 255
    col = col+YG

    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = [int(255*i/GC) for i in range(GC)]
    col = col+GC

    colorwheel[col:col+CB, 1] = [255-int(255*i/CB) for i in range(CB)]
    colorwheel[col:col+CB, 2] = 255
    col = col+CB

    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = [int(255*i/BM) for i in range(BM)]
    col = col+BM

    colorwheel[col:col+MR, 2] = [255-int(255*i/MR) for i in range(MR)]
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def flow2image_normal(flow):
    UMAX = 20.0
    VMAX = 20.0
    PI = np.pi

    height, width, nchannels = flow.shape
    image = np.zeros((height, width, 3))

    colorwheel = set_colorwheel()
    ncols = colorwheel.shape[0]

    u = flow[:,:,0]
    v = flow[:,:,1]

    nan_idx = np.logical_or(np.isnan(u), np.isnan(v))
    unk_idx = np.logical_or(abs(u)>1e4, abs(v)>1e4)

    u[nan_idx==1] = 0
    v[nan_idx==1] = 0
    u[unk_idx] = 0
    v[unk_idx] = 0

    rad = np.sqrt(u*u+v*v)
    maxrad = rad.max()

    u = u / UMAX
    v = v / VMAX
    rad = np.sqrt(u*u+v*v)

    a = np.arctan2(-u, -v)/PI

    fk = (a+1)/2 *(ncols-1)
    k0 = np.floor(fk).astype(int)
    k1 = k0+1

    k1[np.where(k1==ncols)] = 0
    f = fk-k0.astype(float)

    for i in range(colorwheel.shape[1]):
        col = colorwheel[:,i]
        chann0 = col[k0].astype(float)/255.0
        chann1 = col[k1].astype(float)/255.0

        chann = (1-f)*chann0 + f*chann1

        idx0 = np.where(chann<=1)
        idx1 = np.where(chann>1)
        chann[idx0] = 1-rad[idx0]*(1-chann[idx0])
        chann[idx1] = 0.75*chann[idx1]

        chann[unk_idx] = 0

        #image[:,:,i] = np.floor(255*chann*(1-nan_idx))
        image[:,:,i] = chann*(1-nan_idx)

    return image
