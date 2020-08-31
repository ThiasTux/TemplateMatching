import ctypes
import os
from ctypes import *

import numpy as np


class WLCSSCudaParamsTraining:
    """Class for WLCSSCuda computation for parameters training."""

    def __init__(self, templates, streams, num_individuals=1, use_encoding=False):
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
        if not use_encoding:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/training/lib_wlcss_cuda_params_training_optimized.so"))
        elif use_encoding == '2d':
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/lib_wlcss_cuda_params_training_optimized_2d_enc.so"))
        elif use_encoding == '3d':
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/lib_wlcss_cuda_params_training_optimized_3d_enc.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32)]
        self._wlcss_cuda_freemem = wlcss_dll.wlcss_freemem

        self.h_t = templates
        self.h_s = streams

        self.num_templates = len(self.h_t)  # Num block on X
        self.num_streams = len(self.h_s)  # Num block on Y
        self.num_params_sets = num_individuals  # Num thread per block

        self.h_tlen = np.array([len(t) for t in self.h_t]).astype(np.int32)
        self.h_toffsets = np.cumsum(self.h_tlen).astype(np.int32)
        self.h_toffsets = np.insert(self.h_toffsets[0:-1], 0, 0)

        self.h_slen = np.array([len(s) for s in self.h_s]).astype(np.int32)
        self.h_soffsets = np.cumsum(self.h_slen).astype(np.int32)
        self.h_soffsets = np.insert(self.h_soffsets[0:-1], 0, 0)

        # Template as numpy array
        h_ts = np.array([item for sublist in self.h_t for item in sublist]).astype(np.int32)
        # Stream as numpy array
        h_ss = np.array([item for sublist in self.h_s for item in sublist]).astype(np.int32)

        self.h_tmp_windows = np.zeros(
            [(len(h_ts) + 2 * self.num_templates) * self.num_params_sets * self.num_streams]).astype(
            np.int32)
        h_tmp_windows_len = [t + 2 for _ in range(self.num_params_sets) for t in self.h_tlen for _ in
                             range(self.num_streams)]
        # h_tmp_windows_len = np.tile(np.array([t+2 for t in self.h_tlen]), self.num_params_sets * self.num_streams)
        self.h_tmp_windows_offsets = np.cumsum(h_tmp_windows_len).astype(np.int32)
        self.h_tmp_windows_offsets = np.insert(self.h_tmp_windows_offsets[0:-1], 0, 0)

        self.h_mss = np.zeros((len(h_ss) * self.num_params_sets * self.num_templates)).astype(np.int32)
        h_mss_offsets = np.cumsum(np.tile(self.h_slen, self.num_params_sets * self.num_templates)).astype(np.int32)
        self.h_mss_offsets = np.insert(h_mss_offsets, 0, 0)
        self._wlcss_init(self.h_tmp_windows_offsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                         h_ts.ctypes.data_as(POINTER(c_int32)), h_ss.ctypes.data_as(POINTER(c_int32)),
                         self.h_tlen.ctypes.data_as(POINTER(c_int32)), self.h_toffsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_slen.ctypes.data_as(POINTER(c_int32)), self.h_soffsets.ctypes.data_as(POINTER(c_int32)),
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
                         self.h_mss.ctypes.data_as(POINTER(c_int32)),
                         self.h_tmp_windows.ctypes.data_as(POINTER(c_int32)))
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
        if not use_encoding:
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/lib_wlcss_cuda_templates_training_optimized.so"))
        elif use_encoding == '2d':
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/lib_wlcss_cuda_templates_training_optimized_2d_enc.so"))
        elif use_encoding == '3d':
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/lib_wlcss_cuda_templates_training_optimized_3d_enc.so"))
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
        h_ss = np.array([item for sublist in self.h_s for item in sublist]).astype(np.int32)

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
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/training/lib_wlcss_cuda_training_optimized.so"))
        else:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/training/lib_wlcss_cuda_training_optimized.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32)]
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
        h_ss = np.array([item for sublist in self.h_s for item in sublist]).astype(np.int32)

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
    def __init__(self, templates, streams, num_individuals=1, use_encoding=False):
        if not use_encoding:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/training/lib_wlcss_cuda_params_training_optimized.so"))
        elif use_encoding == '2d':
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/lib_wlcss_cuda_params_training_optimized_2d_enc.so"))
        elif use_encoding == '3d':
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/lib_wlcss_cuda_params_training_optimized_3d_enc.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32)]
        self._wlcss_cuda_freemem = wlcss_dll.wlcss_freemem

        self.h_t = templates
        self.h_s = streams

        self.num_templates = len(self.h_t)  # Num block on X
        self.num_streams = len(self.h_s)  # Num block on Y
        self.num_params_sets = num_individuals  # Num thread per block

        self.h_tlen = np.array([len(t) for t in self.h_t]).astype(np.int32)
        self.h_toffsets = np.cumsum(self.h_tlen).astype(np.int32)
        self.h_toffsets = np.insert(self.h_toffsets[0:-1], 0, 0)

        self.h_slen = np.array([len(s) for s in self.h_s]).astype(np.int32)
        self.h_soffsets = np.cumsum(self.h_slen).astype(np.int32)
        self.h_soffsets = np.insert(self.h_soffsets[0:-1], 0, 0)

        # Template as numpy array
        h_ts = np.array([item for sublist in self.h_t for item in sublist]).astype(np.int32)
        # Stream as numpy array
        h_ss = np.array([item for sublist in self.h_s for item in sublist]).astype(np.int32)

        self.h_tmp_windows = np.zeros(
            [(len(h_ts) + 2 * self.num_templates) * self.num_params_sets * self.num_streams]).astype(
            np.int32)
        h_tmp_windows_len = [t + 2 for _ in range(self.num_params_sets) for t in self.h_tlen for _ in
                             range(self.num_streams)]
        # h_tmp_windows_len = np.tile(np.array([t+2 for t in self.h_tlen]), self.num_params_sets * self.num_streams)
        self.h_tmp_windows_offsets = np.cumsum(h_tmp_windows_len).astype(np.int32)
        self.h_tmp_windows_offsets = np.insert(self.h_tmp_windows_offsets[0:-1], 0, 0)

        self.h_mss = np.zeros((len(h_ss) * self.num_params_sets * self.num_templates)).astype(np.int32)
        h_mss_offsets = np.cumsum(np.tile(self.h_slen, self.num_params_sets * self.num_templates)).astype(np.int32)
        self.h_mss_offsets = np.insert(h_mss_offsets, 0, 0)
        self._wlcss_init(self.h_tmp_windows_offsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                         h_ts.ctypes.data_as(POINTER(c_int32)), h_ss.ctypes.data_as(POINTER(c_int32)),
                         self.h_tlen.ctypes.data_as(POINTER(c_int32)), self.h_toffsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_slen.ctypes.data_as(POINTER(c_int32)), self.h_soffsets.ctypes.data_as(POINTER(c_int32)),
                         int(self.num_templates), int(self.num_streams), int(self.num_params_sets), int(len(h_ts)),
                         int(len(h_ss)),
                         int(len(self.h_mss)))

    def compute_wlcss(self, parameters):
        h_params = np.array(parameters).astype(np.int32)
        self._wlcss_cuda(h_params.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss.ctypes.data_as(POINTER(c_int32)),
                         self.h_tmp_windows.ctypes.data_as(POINTER(c_int32)))
        tmp_mss = self.h_mss.reshape((self.num_templates, int(len(self.h_mss) / self.num_templates)))
        return tmp_mss.T

    def cuda_freemem(self):
        self._wlcss_cuda_freemem()


class WLCSSCuda:
    def __init__(self, templates, streams, parameters, use_encoding):
        if not use_encoding:
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/matching/lib_wlcss_cuda.so"))
        elif use_encoding == '2d':
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/matching/lib_wlcss_cuda_2d.so"))
        elif use_encoding == '3d':
            wlcss_dll = ctypes.CDLL(os.path.abspath("libs/cuda/matching/lib_wlcss_cuda_3d.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32)]
        self._wlcss_cuda_freemem = wlcss_dll.wlcss_freemem

        self.h_t = templates
        self.h_s = streams
        s = np.array([parameters]).shape
        if s == tuple((1, 3)):
            self.h_p = [parameters for _ in templates]
        else:
            self.h_p = parameters

        self.num_templates = len(self.h_t)  # Num block on X
        self.num_streams = len(self.h_s)  # Num block on Y
        self.num_params_sets = len(self.h_p)  # Num thread per block

        self.h_tlen = np.array([len(t) for t in self.h_t]).astype(np.int32)
        self.h_toffsets = np.cumsum(self.h_tlen).astype(np.int32)
        self.h_toffsets = np.insert(self.h_toffsets[0:-1], 0, 0)

        self.h_slen = np.array([len(s) for s in self.h_s]).astype(np.int32)
        self.h_soffsets = np.cumsum(self.h_slen).astype(np.int32)
        self.h_soffsets = np.insert(self.h_soffsets[0:-1], 0, 0)

        # Template as numpy array
        self.h_ts = np.array([item for sublist in self.h_t for item in sublist]).astype(np.int32)
        # Stream as numpy array
        self.h_ss = np.array([item for sublist in self.h_s for item in sublist]).astype(np.int32)
        # Params as numpy array
        self.h_params = np.array(parameters).astype(np.int32)

        self.h_tmp_windows = np.zeros(
            [(len(self.h_ts) + 2 * self.num_templates) * self.num_params_sets * self.num_streams]).astype(
            np.int32)
        h_tmp_windows_len = [t + 2 for _ in range(self.num_params_sets) for t in self.h_tlen for _ in
                             range(self.num_streams)]
        # h_tmp_windows_len = np.tile(np.array([t+2 for t in self.h_tlen]), self.num_params_sets * self.num_streams)
        self.h_tmp_windows_offsets = np.cumsum(h_tmp_windows_len).astype(np.int32)
        self.h_tmp_windows_offsets = np.insert(self.h_tmp_windows_offsets[0:-1], 0, 0)

        self.h_mss = np.zeros((len(self.h_ss) * self.num_params_sets * self.num_templates)).astype(np.int32)
        h_mss_offsets = np.cumsum(np.tile(self.h_slen, self.num_params_sets * self.num_templates)).astype(np.int32)
        self.h_mss_offsets = np.insert(h_mss_offsets, 0, 0)
        self._wlcss_init(self.h_tmp_windows_offsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_ts.ctypes.data_as(POINTER(c_int32)), self.h_ss.ctypes.data_as(POINTER(c_int32)),
                         self.h_tlen.ctypes.data_as(POINTER(c_int32)), self.h_toffsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_slen.ctypes.data_as(POINTER(c_int32)), self.h_soffsets.ctypes.data_as(POINTER(c_int32)),
                         int(self.num_templates), int(self.num_streams), int(self.num_params_sets), int(len(self.h_ts)),
                         int(len(self.h_ss)),
                         int(len(self.h_mss)))

    def compute_wlcss(self):
        self._wlcss_cuda(self.h_params.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss.ctypes.data_as(POINTER(c_int32)),
                         self.h_tmp_windows.ctypes.data_as(POINTER(c_int32)))
        tmp_mss = np.array([self.h_mss[offset - 1] for offset in self.h_mss_offsets[1:]])
        mss = [np.reshape(np.ravel(x), (self.num_streams, self.num_templates), order='F') for x in
               np.reshape(tmp_mss, (self.num_params_sets, self.num_streams, self.num_templates))]
        new_mss = np.zeros(mss[0].shape)
        for i in range(new_mss.shape[1]):
            new_mss[:, i] = mss[i][:, i]
        return new_mss

    def cuda_freemem(self):
        self._wlcss_cuda_freemem()
