import ctypes
import os
from ctypes import *

import numpy as np


class WLCSSCudaParamsTraining:
    """Class for WLCSSCuda computation for parameters training."""

    def __init__(self, templates, streams, num_individuals, use_encoding):
        """
        Initialization of WLCSSCuda for parameters optimization.
        :param templates: list
            list of templates (numpy.ndarray) with time,data,label,user
        :param streams: list
            list of streams (numpy.ndarray) with time,data,label,user
        :param num_individuals: int
            num of parameters set
        :param use_encoding: boolean
            Use encoding for computing the distance
        """
        if use_encoding:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/lib_wlcss_cuda_params_training_optimized.so"))
        else:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/lib_wlcss_cuda_params_training_optimized.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32)]
        self._wlcss_cuda_freemem = wlcss_dll.wlcss_freemem

        self.h_t = templates
        self.h_s = streams

        self.num_templates = len(self.h_t)  # Num block on X
        self.num_streams = len(self.h_s)  # Num block on Y
        self.num_params_sets = num_individuals  # Num thread per block

        h_tlen = np.array([len(t) for t in self.h_t]).astype(np.int32)
        h_toffsets = np.cumsum(h_tlen).astype(np.int32)
        h_toffsets = np.insert(h_toffsets[0:-1], 0, 0)

        h_slen = np.array([len(s) for s in self.h_s]).astype(np.int32)
        h_soffsets = np.cumsum(h_slen).astype(np.int32)
        h_soffsets = np.insert(h_soffsets[0:-1], 0, 0)

        # Template as numpy array
        h_ts = np.array([item for sublist in self.h_t for item in sublist[:, 1]]).astype(np.int32)
        # Stream as numpy array
        h_ss = np.array([item for sublist in self.h_s for item in sublist[:, 1]]).astype(np.int32)

        self.h_mss = np.zeros((len(h_ss) * self.num_params_sets * self.num_templates)).astype(np.int32)
        h_mss_offsets = np.cumsum(np.tile(h_slen, self.num_params_sets * self.num_templates)).astype(np.int32)
        self.h_mss_offsets = np.insert(h_mss_offsets, 0, 0)
        self._wlcss_init(self.h_mss.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                         h_ts.ctypes.data_as(POINTER(c_int32)), h_ss.ctypes.data_as(POINTER(c_int32)),
                         h_tlen.ctypes.data_as(POINTER(c_int32)), h_toffsets.ctypes.data_as(POINTER(c_int32)),
                         h_slen.ctypes.data_as(POINTER(c_int32)), h_soffsets.ctypes.data_as(POINTER(c_int32)),
                         int(self.num_templates), int(self.num_streams), int(self.num_params_sets), int(len(h_ts)),
                         int(len(h_ss)),
                         int(len(self.h_mss)))

    def compute_wlcss(self, parameters):
        """
        Compute the scores between templates and streams, for each parameters set.
        :param parameters: list
            list of parameters set to be evaluated
        :return: list,
            list of numpy.ndarray with the last line of the matching scores between templates and streams
        """
        h_params = np.array(parameters).astype(np.int32)
        self._wlcss_cuda(h_params.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss.ctypes.data_as(POINTER(c_int32)))
        tmp_mss = np.array([self.h_mss[offset - 1] for offset in self.h_mss_offsets[1:]])
        mss = [np.reshape(np.ravel(x), (self.num_streams, self.num_templates), order='F') for x in
               np.reshape(tmp_mss, (self.num_params_sets, self.num_streams, self.num_templates))]
        return mss

    def cuda_freemem(self):
        """
        Free memory of CUDA
        """
        self._wlcss_cuda_freemem()


class WLCSSCudaTemplatesTraining:
    """Class for WLCSSCuda computation for templates training."""

    def __init__(self, streams, params, t_len, num_individuals, use_encoding):
        """
        Initialization of WLCSSCuda for template optimization.
        :param streams: list
            list of streams (numpy.ndarray) with time,data,label,user
        :param params: list
            list of parameters set
        :param t_len: int
            length of the template
        :param num_individuals: int
            num of templates to be evaluated
        :param use_encoding: boolean
            Use encoding for computing the distance
        """
        if use_encoding:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/lib_wlcss_cuda_templates_training_optimized.so"))
        else:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/lib_wlcss_cuda_templates_training_optimized.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32)]
        self._wlcss_cuda_freemem = wlcss_dll.wlcss_freemem

        self.h_params = np.array(params).astype(np.int32)
        self.h_s = streams

        h_tlen = np.array([t_len for t in range(num_individuals)]).astype(np.int32)
        h_toffsets = np.cumsum(h_tlen).astype(np.int32)
        h_toffsets = np.insert(h_toffsets[0:-1], 0, 0)

        self.num_templates = num_individuals  # Num block on X
        self.num_streams = len(self.h_s)  # Num block on Y
        self.num_params_sets = 1  # Num thread per block

        h_slen = np.array([len(s) for s in self.h_s]).astype(np.int32)
        h_soffsets = np.cumsum(h_slen).astype(np.int32)
        h_soffsets = np.insert(h_soffsets[0:-1], 0, 0)

        # Stream as numpy array
        h_ss = np.array([item for sublist in self.h_s for item in sublist[:, 1]]).astype(np.int32)

        self.h_mss = np.zeros((len(h_ss) * self.num_params_sets * self.num_templates)).astype(np.int32)
        h_mss_offsets = np.cumsum(np.tile(h_slen, self.num_params_sets * self.num_templates)).astype(np.int32)
        self.h_mss_offsets = np.insert(h_mss_offsets, 0, 0)
        self._wlcss_init(self.h_mss.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                         h_ss.ctypes.data_as(POINTER(c_int32)), h_slen.ctypes.data_as(POINTER(c_int32)),
                         h_soffsets.ctypes.data_as(POINTER(c_int32)),
                         h_tlen.ctypes.data_as(POINTER(c_int32)), h_toffsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_params.ctypes.data_as(POINTER(c_int32)),
                         int(self.num_templates), int(self.num_streams), int(self.num_params_sets),
                         int(t_len * num_individuals),
                         int(len(h_ss)),
                         int(len(self.h_mss)))

    def compute_wlcss(self, templates):
        """
        Compute the scores between templates and streams
        :param templates: list
            list of templates
        :return: list of numpy.ndarray with the last line of the matching scores between templates and streams
        """
        h_ts = np.array([item for sublist in templates for item in sublist]).astype(np.int32)
        self._wlcss_cuda(h_ts.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss.ctypes.data_as(POINTER(c_int32)))
        tmp_mss = np.array([self.h_mss[offset - 1] for offset in self.h_mss_offsets[1:]])
        mss = [np.reshape(np.ravel(x), (self.num_streams, self.num_templates), order='F') for x in
               np.reshape(tmp_mss, (self.num_params_sets, self.num_streams, self.num_templates))]
        return mss

    def cuda_freemem(self):
        """
        Free memory of CUDA
        """
        self._wlcss_cuda_freemem()


class WLCSSCudaTraining:
    """Class for WLCSSCuda computation for complete training."""

    def __init__(self, streams, t_len, num_individuals, use_encoding):
        """
        Initialization of WLCSSCuda for template optimization.
        :param streams: list
            list of streams (numpy.ndarray) with time,data,label,user
        :param params: list
            list of parameters set
        :param t_len: int
            length of the template
        :param num_individuals: int
            num of templates to be evaluated
        :param use_encoding: boolean
            Use encoding for computing the distance
        """
        if use_encoding:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/lib_wlcss_cuda_training_optimized.so"))
        else:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/lib_wlcss_cuda_training_optimized.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32)]
        self._wlcss_cuda_freemem = wlcss_dll.wlcss_freemem

        self.h_s = streams

        h_tlen = np.array([t_len for t in range(num_individuals)]).astype(np.int32)
        h_toffsets = np.cumsum(h_tlen).astype(np.int32)
        h_toffsets = np.insert(h_toffsets[0:-1], 0, 0)

        self.num_templates = num_individuals  # Num block on X
        self.num_streams = len(self.h_s)  # Num block on Y
        self.num_params_sets = 1  # Num thread per block

        h_slen = np.array([len(s) for s in self.h_s]).astype(np.int32)
        h_soffsets = np.cumsum(h_slen).astype(np.int32)
        h_soffsets = np.insert(h_soffsets[0:-1], 0, 0)

        # Stream as numpy array
        h_ss = np.array([item for sublist in self.h_s for item in sublist[:, 1]]).astype(np.int32)

        self.h_mss = np.zeros((len(h_ss) * self.num_params_sets * self.num_templates)).astype(np.int32)
        h_mss_offsets = np.cumsum(np.tile(h_slen, self.num_params_sets * self.num_templates)).astype(np.int32)
        self.h_mss_offsets = np.insert(h_mss_offsets, 0, 0)
        self._wlcss_init(self.h_mss.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                         h_ss.ctypes.data_as(POINTER(c_int32)), h_slen.ctypes.data_as(POINTER(c_int32)),
                         h_soffsets.ctypes.data_as(POINTER(c_int32)),
                         h_tlen.ctypes.data_as(POINTER(c_int32)), h_toffsets.ctypes.data_as(POINTER(c_int32)),
                         int(self.num_templates), int(self.num_streams), int(self.num_params_sets),
                         int(t_len * num_individuals),
                         int(len(h_ss)),
                         int(len(self.h_mss)))

    def compute_wlcss(self, parameters, templates):
        """
        Compute the scores between templates and streams
        :param templates: list
            list of templates
        :return: list of numpy.ndarray with the last line of the matching scores between templates and streams
        """
        h_params = np.array(parameters).astype(np.int32)
        h_ts = np.array([item for sublist in templates for item in sublist]).astype(np.int32)
        self._wlcss_cuda(h_params.ctypes.data_as(POINTER(c_int32)),
                         h_ts.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss.ctypes.data_as(POINTER(c_int32)))
        tmp_mss = np.array([self.h_mss[offset - 1] for offset in self.h_mss_offsets[1:]])
        mss = [np.reshape(np.ravel(x), (self.num_streams, self.num_templates), order='F') for x in
               np.reshape(tmp_mss, (self.num_params_sets, self.num_streams, self.num_templates))]
        return mss

    def cuda_freemem(self):
        """
        Free memory of CUDA
        """
        self._wlcss_cuda_freemem()


class WLCSSCudaContinuous:
    def __init__(self, templates, streams, num_individuals, use_encoding):
        if use_encoding:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/lib_wlcss_cuda_params_training_optimized.so"))
        else:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/lib_wlcss_cuda_params_training_optimized.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32)]
        self._wlcss_cuda_freemem = wlcss_dll.wlcss_freemem

        self.h_t = templates
        self.h_s = streams

        self.num_templates = len(self.h_t)  # Num block on X
        self.num_streams = len(self.h_s)  # Num block on Y
        self.num_params_sets = num_individuals  # Num thread per block

        h_tlen = np.array([len(t) for t in self.h_t]).astype(np.int32)
        h_toffsets = np.cumsum(h_tlen).astype(np.int32)
        h_toffsets = np.insert(h_toffsets[0:-1], 0, 0)

        h_slen = np.array([len(s) for s in self.h_s]).astype(np.int32)
        h_soffsets = np.cumsum(h_slen).astype(np.int32)
        h_soffsets = np.insert(h_soffsets[0:-1], 0, 0)

        # Template as numpy array
        h_ts = np.array([item for sublist in self.h_t for item in sublist[:, 1]]).astype(np.int32)
        # Stream as numpy array
        h_ss = np.array([item for sublist in self.h_s for item in sublist[:, 1]]).astype(np.int32)

        self.h_mss = np.zeros((len(h_ss) * self.num_params_sets * self.num_templates)).astype(np.int32)
        h_mss_offsets = np.cumsum(np.tile(h_slen, self.num_params_sets * self.num_templates)).astype(np.int32)
        self.h_mss_offsets = np.insert(h_mss_offsets, 0, 0)
        self._wlcss_init(self.h_mss.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                         h_ts.ctypes.data_as(POINTER(c_int32)), h_ss.ctypes.data_as(POINTER(c_int32)),
                         h_tlen.ctypes.data_as(POINTER(c_int32)), h_toffsets.ctypes.data_as(POINTER(c_int32)),
                         h_slen.ctypes.data_as(POINTER(c_int32)), h_soffsets.ctypes.data_as(POINTER(c_int32)),
                         int(self.num_templates), int(self.num_streams), int(self.num_params_sets), int(len(h_ts)),
                         int(len(h_ss)),
                         int(len(self.h_mss)))

    def compute_wlcss(self, parameters):
        h_params = np.array(parameters).astype(np.int32)
        self._wlcss_cuda(h_params.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss.ctypes.data_as(POINTER(c_int32)))
        tmp_mss = self.h_mss.reshape((self.num_templates, int(len(self.h_mss) / self.num_templates)))
        return tmp_mss.T

    def cuda_freemem(self):
        self._wlcss_cuda_freemem()
