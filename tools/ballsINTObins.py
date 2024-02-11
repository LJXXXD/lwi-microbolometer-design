
import numpy as np


def ballsINTObins(num_balls, num_bins):

    if num_balls==0:
        return np.zeros((1, num_bins))
    
    if num_bins==1:
        return np.asarray([num_balls])
    
    all_rest = []
    for i in range(num_balls+1):
        rest = ballsINTObins(num_balls-i, num_bins-1)
        all_rest.append(np.c_[np.ones(rest.shape[0])*i, rest])
    
    result = np.concatenate(all_rest, axis=0)
    return result