import sys 
import os
import numpy as np
import os.path as osp 
import scipy.io as sio 
from copy import copy
from PIL import Image
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def sortbyvar(data):
    sortdata = np.zeros(data.shape)
    sumdata = np.sum(data, axis=3)
    flat = sumdata.reshape(sumdata.shape[0], sumdata.shape[1]*sumdata.shape[2])
    std = np.std(flat, axis=1)
    order = np.argsort(std)
    for i in range(order.shape[0]):
        sortdata[i,:,:,:] = data[order[i],:,:,:]
    return sortdata

def vis_square(data, fname):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    data = np.squeeze(data)
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    data = sortbyvar(data)   
 
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, 96 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((6, 16) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((6 * data.shape[1], 16 * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    plt.savefig(fname,bbox_inches='tight')

if __name__ == '__main__':
    """
    usage: visualize.py model output_fig
    """


    caffe_root = '/home/hylee/Unsupervised/caffe_s2/'
    sys.path.append(caffe_root + 'python')
    import caffe
    gpu_id = 0
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    caffemodel = sys.argv[1]
    fname = sys.argv[2]
    prototxt = 'prototxt/deploy_RGB.prototxt'

    conv1 = 'conv1'
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    filters = net.params[conv1][0].data
    vis_square(filters.transpose(0, 2, 3, 1), fname)
