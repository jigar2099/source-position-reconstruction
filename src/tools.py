from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from PIL import Image
from src.surrounding import surrounding

def data_merger(src_dir, label_dir, data_set):
    train_x = []
    train_y = []
    for i in glob.glob(src_dir+"*"+data_set+"*"):
        src_split1 = i.split('_SRC_')[1]
        src_split0 = i.split('_SRC_')[0]
        src_split2 = src_split0.split('/')[-1]        
        y_path = label_dir+src_split2+'_LABEL_'+src_split1
        x = np.load(i)
        y = np.load(y_path)
        train_x.append(x)
        train_y.append(y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = train_x.reshape(train_x.shape[0]*train_x.shape[1],train_x.shape[2],train_x.shape[3])
    train_y = train_y.reshape(train_y.shape[0]*train_y.shape[1],train_y.shape[2],train_y.shape[3])   
    return train_x, train_y

def multiple_data(path_list,type_of_data):
    train_x = []
    train_y = []
    for i in path_list[:]:
        src_dir = "./src/"+i+"/"; label_dir = "./labels/"+i+"/"
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