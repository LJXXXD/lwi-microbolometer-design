
import numpy as np


def ballsINTObins(m, n):

    if m==0:
        return np.zeros((1, n))
    
    if n==1:
        return np.asarray([m])
    
    all_rest = []
    for i in range(m+1):
        rest = ballsINTObins(m-i, n-1)
        all_rest.append(np.c_[np.ones(rest.shape[0])*i, rest])
    
    result = np.concatenate(all_rest, axis=0)
    return result