import ctypes as ct
import os

import numpy as np

_wlcss = ct.CDLL(os.path.abspath("libs/libwlcss.so"))

_wlcss.wlcss_init.argtypes = [ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32]
_wlcss.wlcss.argtypes = [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]
_wlcss.wlcss.restype = ct.POINTER(ct.c_int32)
_wlcss.free_mem.argtypes = [ct.POINTER(ct.c_int32)]

penalty, reward, acceptdist, lent, lens = None, None, None, None, None


def wlcss_init(p, r, a, nt, ns):
    _wlcss.wlcss_init(ct.c_int32(p), ct.c_int32(r), ct.c_int32(a), ct.c_int32(nt), ct.c_int32(ns))


def compute_wlcss(t, s, p, r, a):
    t = np.array(t, dtype=np.int32)
    s = np.array(s, dtype=np.int32)
    p_t = t.ctypes.data_as(ct.POINTER(ct.c_int32))
    p_s = s.ctypes.data_as(ct.POINTER(ct.c_int32))
    nt = len(t)
    ns = len(s)
    _wlcss.wlcss_init(ct.c_int32(p), ct.c_int32(r), ct.c_int32(a), ct.c_int32(nt), ct.c_int32(ns))
    p_mc = _wlcss.wlcss(p_t, p_s)
    mc = np.array(np.fromiter(p_mc, dtype=np.int32, count=(nt + 1) * (ns + 1)))
    matching_cost = np.reshape(mc, [nt + 1, ns + 1])
    _wlcss.free_mem(p_mc)
    return matching_cost[-1][-1], matching_cost
