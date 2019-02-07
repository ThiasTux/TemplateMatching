import ctypes
from ctypes import *

import numpy as np


def get_wlcss_cuda():
    dll = ctypes.CDLL("libs/cuda/lib_wlcss_cuda.so")
    func = dll.wlcss_cuda
    func.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                    POINTER(c_int32),
                    c_int, c_int, c_int, c_int, c_int, c_int]
    return func


def wlcss_cuda(templates, streams, parameters, use_encoding=False):
    _wlcss_cuda = get_wlcss_cuda()

    h_t = templates
    h_s = streams
    h_params = parameters

    num_templates = len(h_t)  # Num block on X
    num_streams = len(h_s)  # Num block on Y
    num_params_sets = len(h_params)  # Num thread per block

    h_tlen = np.array([len(t) for t in h_t]).astype(np.int32)
    h_toffsets = np.cumsum(h_tlen).astype(np.int32)
    h_toffsets = np.insert(h_toffsets[0:-1], 0, 0)

    h_slen = np.array([len(s) for s in h_s]).astype(np.int32)
    h_soffsets = np.cumsum(h_slen).astype(np.int32)
    h_soffsets = np.insert(h_soffsets[0:-1], 0, 0)

    h_ts = np.array([item for sublist in h_t for item in sublist]).astype(np.int32)  # Template as numpy array
    h_ss = np.array([item for sublist in h_s for item in sublist]).astype(np.int32)  # Stream as numpy array

    h_mss = np.zeros((len(h_ss) * num_params_sets * num_templates)).astype(np.int32)
    h_mss_offsets = np.cumsum(np.tile(h_slen, num_params_sets * num_templates)).astype(np.int32)
    h_mss_offsets = np.insert(h_mss_offsets[0:-1], 0, 0)
    print(h_mss_offsets)

    _wlcss_cuda(h_mss.ctypes.data_as(POINTER(c_int32)), h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                h_ts.ctypes.data_as(POINTER(c_int32)), h_ss.ctypes.data_as(POINTER(c_int32)),
                h_tlen.ctypes.data_as(POINTER(c_int32)), h_toffsets.ctypes.data_as(POINTER(c_int32)),
                h_slen.ctypes.data_as(POINTER(c_int32)), h_soffsets.ctypes.data_as(POINTER(c_int32)),
                h_params.ctypes.data_as(POINTER(c_int32)), int(num_templates),
                int(num_streams), int(num_params_sets), int(len(h_ts)), int(len(h_ss)), int(len(h_mss)))

    tmp_mss = np.array([h_mss[offset - 1] for offset in h_mss_offsets[1:]])
    mss = [np.reshape(np.ravel(x), (num_streams, num_templates), order='F') for x in
           np.reshape(tmp_mss, (num_params_sets, num_streams, num_templates))]
    return mss
