from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from PIL import Image
from src.surrounding import surrounding

def data_merger(dataset_type='train', data_type='src', noise_type='all', dirs = ['noise_7e-3']):
    X = []
    for h in dirs:
        for i in glob.glob('../DATASETS/'):
            for j in glob.glob(i+'*'):
                adjusted_j = i+h
                for k in glob.glob(adjusted_j+'/*'):
                    adjusted_k = k.split('\ ')[0].split('\\')
                    if adjusted_k[1]==data_type:
                        src_path = adjusted_k[0]+'/'+adjusted_k[1]+'/'
                        for l in glob.glob(src_path+'*'):
                            if l.split('_')[4] == dataset_type:
                                all_l = l.split('\ ')[0].split('\\')[0]+'/'+l.split('\ ')[0].split('\\')[1]
                                if all_l.split('_')[3]==noise_type:
                                    x = np.load(all_l)
                                    X.append(x)
                                elif noise_type == 'all':
                                    x = np.load(all_l)
                                    X.append(x)
                                else: pass
                            else: pass
                    else: pass
    X = np.array(X)
    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3])
    return X

def multiple_data(path_list,type_of_data):
    train_x = []
    train_y = []
    for i in path_list[:]:
        src_dir = "/..DATASETS/"+i+'/src/'; label_dir = "/..DATASETS/"+i+'/labels/'
        data_set = type_of_data
        x1, y1 = data_merger(src_dir, label_dir, type_of_data)
        train_x.append(x1)
        train_y.append(y1)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = train_x.reshape(train_x.shape[0]*train_x.shape[1],train_x.shape[2],train_x.shape[3])
    train_y = train_y.reshape(train_y.shape[0]*train_y.shape[1],train_y.shape[2],train_y.shape[3])
    return train_x,train_y


def get_indices_of_ones(arr, src_num):
    """
    Returns the indices of ones appearing in the input numpy array.

    Parameters:
    arr (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: A numpy array of indices where the input array has ones.
    """
    return np.argwhere(arr == src_num)

def radius_eda(samp, pred, radius=1, src_num=1):
    
    '''
    This function takes 2d sample and respected prediction as "samp" and "pred".
    The user need to define the desired radius, e.g, 1,2,3
    The user need to define the source number to be considered, either
    single source or multiple sources
    '''
    
    ind_of_src_num = get_indices_of_ones(samp, src_num)
    TOT_SRC = []; MEAN_ERR = []; STD_ERR = []
    for i in ind_of_src_num:
        y = surrounding(samp,i,radius=radius, fill=0)
        totl_src = np.sum(y.flatten())
        TOT_SRC.append(totl_src)
        py = surrounding(pred, i, radius=radius, fill=0)
        error = y-py
        mean_error = np.mean(error.flatten())
        MEAN_ERR.append(mean_error)
        std_error = np.std(error.flatten())
        STD_ERR.append(std_error)
    return TOT_SRC, MEAN_ERR, STD_ERR
#samp_idx = np.random.randint(55000,59759,1)[0]
#samp = test_y[samp_idx]
#pred = pred_y[samp_idx]
#print(samp.shape)
#radius_eda(samp, pred, radius=2,src_num=1)