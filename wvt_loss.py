import pywt
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.stats as sp


def wavelet(inp_tsr):

    w = pywt.Wavelet('haar')
    a1 = np.expand_dims(w.dec_lo, axis = -1)
    a2 = np.expand_dims(w.dec_lo, axis = 0)
    a = np.dot(a1,a2)
    a = np.expand_dims(a,axis = -1)
    a = np.expand_dims(a,axis = -1)

    dh = np.expand_dims(w.dec_hi, axis = 0)
    ay = np.expand_dims(w.dec_lo, axis = -1)
    grad_h = np.dot(ay,dh)
    grad_h = np.expand_dims(grad_h, axis = -1)
    grad_h = np.expand_dims(grad_h, axis = -1)

    dy = np.expand_dims(w.dec_hi, axis = -1)
    ah = np.expand_dims(w.dec_lo, axis = 0)
    grad_y = np.dot(dy, ah)
    grad_y = np.expand_dims(grad_y,axis = -1)
    grad_y = np.expand_dims(grad_y, axis = -1)
    
    dy = np.expand_dims(w.dec_hi, axis = -1)
    dh = np.expand_dims(w.dec_hi, axis = 0)
    grad_d = np.dot(dy, dh)
    grad_d = np.expand_dims(grad_d, axis = -1)
    grad_d = np.expand_dims(grad_d, axis = -1)
    
    wv_fil = np.concatenate((a, grad_h, grad_y, grad_d), axis = -1)
    wv_fil_tsr = tf.convert_to_tensor(wv_fil, dtype = tf.float32)

    lvl1 = tf.nn.conv2d(inp_tsr, wv_fil_tsr, strides = (1,2,2,1), padding = 'SAME')
   
    
    lvl2 = tf.nn.conv2d(tf.expand_dims(lvl1[:,:,:,0], axis = -1), wv_fil_tsr, strides = (1,2,2,1), padding = 'SAME')
    tmp1 = tf.nn.conv2d(tf.expand_dims(lvl1[:,:,:,1], axis = -1), wv_fil_tsr, strides = (1,2,2,1), padding = 'SAME')

    lvl2 = tf.concat([lvl2, tmp1], axis = -1)
    tmp1 = tf.nn.conv2d(tf.expand_dims(lvl1[:,:,:,2], axis = -1), wv_fil_tsr, strides = (1,2,2,1), padding = 'SAME')
    lvl2 = tf.concat([lvl2, tmp1], axis = -1)

    tmp1 = tf.nn.conv2d(tf.expand_dims(lvl1[:,:,:,3], axis = -1), wv_fil_tsr, strides = (1,2,2,1), padding = 'SAME')
    lvl2 = tf.concat([lvl2, tmp1], axis = -1)   
    
    lvl3 = tf.nn.conv2d(tf.expand_dims(lvl2[:,:,:,0], axis = -1), wv_fil_tsr, strides = (1,2,2,1), padding = 'SAME')
    tmp2 = tf.nn.conv2d(tf.expand_dims(lvl2[:,:,:,1], axis = -1), wv_fil_tsr, strides = (1,2,2,1), padding = 'SAME') 
    lvl3 = tf.concat([lvl3, tmp2], axis = -1)

    for j in range(14):
        tmp2 = tf.nn.conv2d(tf.expand_dims(lvl2[:,:,:,j+2], axis = -1), wv_fil_tsr, strides = (1,2,2,1), padding = 'SAME')
        lvl3 = tf.concat([lvl3, tmp2], axis = -1)
        
    return lvl3



def wt_diff(y_true, y_pred):
    w_true = wavelet(y_true)
    w_pred = wavelet(y_pred)
    h = np.arange(64)
    lam = 10*sp.norm.pdf(h, loc=32.5, scale=12.5)
    lam = np.diag(lam)
    lam_tsr = tf.convert_to_tensor(lam, dtype = tf.float32)

    diff = tf.math.squared_difference(w_true, w_pred)
    diff = tf.tensordot(diff, lam_tsr, axes = [[3], [0]])
    diff1 = tf.reduce_mean(diff)
    
    return diff1

