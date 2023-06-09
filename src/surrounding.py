import numpy as np
import pandas as pd
import glob

def surrounding(x, idx, radius=1, fill=0):
    """ 
    Gets surrounding elements from a numpy array 
  
    Parameters: 
    x (ndarray of rank N): Input array
    idx (N-Dimensional Index): The index at which to get surrounding elements. If None is specified for a particular axis,
        the entire axis is returned.
    radius (array-like of rank N or scalar): The radius across each axis. If None is specified for a particular axis, 
        the entire axis is returned.
    fill (scalar or None): The value to fill the array for indices that are out-of-bounds.
        If value is None, only the surrounding indices that are within the original array are returned.
  
    Returns: 
    ndarray: The surrounding elements at the specified index
    """
    
    assert len(idx) == len(x.shape)
    
    if np.isscalar(radius): radius = tuple([radius for i in range(len(x.shape))])
    
    slices = []
    paddings = []
    for axis in range(len(x.shape)):
        if idx[axis] is None or radius[axis] is None:
            slices.append(slice(0, x.shape[axis]))
            paddings.append((0, 0))
            continue
            
        r = radius[axis]
        l = idx[axis] - r 
        r = idx[axis] + r
        
        pl = 0 if l > 0 else abs(l)
        pr = 0 if r < x.shape[axis] else r - x.shape[axis] + 1
        
        slices.append(slice(max(0, l), min(x.shape[axis], r+1)))
        paddings.append((pl, pr))
    
    if fill is None: return x[slices]
    return np.pad(x[slices], paddings, 'constant', constant_values=fill)