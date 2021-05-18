import numpy as np

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_sift_features(feature_file):
    xt = fvecs_read(feature_file)
    print('Loaded features:', xt)
    print('Shape:', xt.shape)

    return xt

# load_sift_features('./siftsmall_groundtruth.ivecs')
