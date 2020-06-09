"""
Test WLCSSCuda
"""
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining
import numpy as np

templates = [np.array([4, 4, 5, 6, 0, 0, 7, 5, 4, 4])]
streams = [np.array([4, 4, 5, 6, 0, 0, 7, 5, 4, 4])]
params = [8, 1, 0]

m_wlcss_cuda = WLCSSCudaParamsTraining(templates, streams, 1, False)
mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
m_wlcss_cuda.cuda_freemem()

print(mss)
