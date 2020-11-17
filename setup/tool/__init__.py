import numpy as np
import torch
import os
import time
import cv2
import sys
from functools import wraps


def syspathinit(path, depth=1):
    CODEPATH = os.path.abspath(path)
    for i in range(depth):
        CODEPATH = os.path.dirname(CODEPATH)
    BASEPATH = os.path.dirname(CODEPATH)

    if CODEPATH not in sys.path:
        sys.path.append(CODEPATH)
    return CODEPATH, BASEPATH


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result

    return measure_time


__all__ = ['np', 'torch', 'os', 'time', 'cv2', 'CODEPATH', 'BASEPATH']
