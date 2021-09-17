import tensorflow as tf 
import os
from keras.utils import multi_gpu_model 
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D
from keras.layers import Flatten, Add
from keras.layers import Concatenate, Activation, Layer
from keras.layers import LeakyReLU, BatchNormalization, Lambda, PReLU, Multiply
from keras.initializers import constant, RandomUniform
import matplotlib.pyplot as plt
import numpy as np
from metrics import metrics
import pickle
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import time
from conv import ComplexConv2D
from bn import ComplexBatchNormalization
from utils import GetReal, GetImag, GetAbs
from keras import backend as K
from tensorflow.python.ops import array_ops

data_path='/home/Co-VeGAN/testing_gt.pickle'
usam_path='/home/Co-VeGAN/testing_usamp_1dg_a5.pickle'

df=open(data_path,'rb')
uf=open(usam_path,'rb')

dataset_real=pickle.load(df)
u_sampled_data=pickle.load(uf)

data = np.asarray(dataset_real[0:2000,:,:], dtype = 'float32')
usp_data = np.expand_dims(u_sampled_data[0:2000,:,:], axis = -1)

inp_shape = (256,256,2)
trainable = False
accel = 5

usp_img = usp_data.imag 
usp_real = usp_data.real 

#to standardize the testing data, use values from the training data
#max_val=1.4005044013171297 #for a3
max_val = 1.4297224443392373 #for a5
#max_val = 360.25993392112275 #for a10
#max_val = 1.4196847643977173 #for a3 radial
#max_val = 359.39426680605976  #for a3 spiral

usp_real = usp_real/max_val
usp_img = usp_img/max_val

data_gen = np.concatenate((usp_real, usp_img), axis =-1)

class sinusoid(Layer):
    def __init__(self, **kwargs):
        super(sinusoid, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.s1 = self.add_weight(name='s1',shape=[1, 1, int(input_shape[3]/2)],initializer = RandomUniform(minval=-0.25, maxval=0.25),trainable=True)
        self.w1 = self.add_weight(name='w1',shape=[1, 1, int(input_shape[3]/2)],initializer = RandomUniform(minval=-0.05, maxval=0.05),trainable=True)
        
        self.s2 = self.add_weight(name='s2',shape=[1, 1, int(input_shape[3]/2)],initializer = RandomUniform(minval=-0.25, maxval=0.25),trainable=True)
        self.w2 = self.add_weight(name='w2',shape=[1, 1, int(input_shape[3]/2)],initializer = RandomUniform(minval=-0.05, maxval=0.05),trainable=True)
        
        self.s3 = self.add_weight(name='s3',shape=[1, 1, int(input_shape[3]/2)],initializer = RandomUniform(minval=-0.25, maxval=0.25),trainable=True)
        self.w3 = self.add_weight(name='w3',shape=[1, 1, int(input_shape[3]/2)],initializer = RandomUniform(minval=-0.05, maxval=0.05),trainable=True)
        
        self.phi = self.add_weight(name='phi',shape=[1, 1, int(input_shape[3]/2)],initializer = RandomUniform(minval=-0.1, maxval=0.1),trainable=True)
        
        self.s1 = tf.keras.backend.repeat_elements(self.s1, rep=input_shape[1], axis=0)
        self.s1 = tf.keras.backend.repeat_elements(self.s1, rep=input_shape[2], axis=1)
        self.w1 = tf.keras.backend.repeat_elements(self.w1, rep=input_shape[1], axis=0)
        self.w1 = tf.keras.backend.repeat_elements(self.w1, rep=input_shape[2], axis=1)
        
        self.s2 = tf.keras.backend.repeat_elements(self.s2, rep=input_shape[1], axis=0)
        self.s2 = tf.keras.backend.repeat_elements(self.s2, rep=input_shape[2], axis=1)
        self.w2 = tf.keras.backend.repeat_elements(self.w2, rep=input_shape[1], axis=0)
        self.w2 = tf.keras.backend.repeat_elements(self.w2, rep=input_shape[2], axis=1)

        self.s3 = tf.keras.backend.repeat_elements(self.s3, rep=input_shape[1], axis=0)
        self.s3 = tf.keras.backend.repeat_elements(self.s3, rep=input_shape[2], axis=1)
        self.w3 = tf.keras.backend.repeat_elements(self.w3, rep=input_shape[1], axis=0)
        self.w3 = tf.keras.backend.repeat_elements(self.w3, rep=input_shape[2], axis=1)
        
        self.phi = tf.keras.backend.repeat_elements(self.phi, rep=input_shape[1], axis=0)
        self.phi = tf.keras.backend.repeat_elements(self.phi, rep=input_shape[2], axis=1)

        super(sinusoid, self).build(input_shape)
        
    def call(self, x):
        real_act = GetReal()(x)
        imag_act = GetImag()(x)
        phase = tf.complex(real_act, imag_act)
        phase = tf.angle(phase)
        phase_new = (self.w1*(1.0 + tf.cos(phase - self.s1)) + self.w2*(1.0 + tf.cos(2.0*(phase - self.s2))) + self.w3*(1.0 + tf.cos(4.0*(phase - self.s3))))/(K.abs(self.w1) + K.abs(self.w2) + K.abs(self.w3) +0.000005)
        phase_new = Lambda(lambda x:x/2)(phase_new)
        mag = GetAbs()(x)
        mag = Multiply()([mag, phase_new])
        phase_new = tf.cos(phase+self.phi)
        phase_new = Lambda(lambda x:x)(phase_new)
        real_act = Multiply()([mag, phase_new])
        phase_new = tf.sin(phase+self.phi)
        phase_new = Lambda(lambda x:x)(phase_new)
        imag_act = Multiply()([mag, phase_new])
        imag_act = K.concatenate([real_act, imag_act], axis=-1)
        return imag_act


def resden(x,fil,gr,beta,gamma_init,trainable):    
    x1=ComplexConv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex',init_criterion='he', bias_initializer = 'zeros')(x)
    x1=ComplexBatchNormalization()(x1)
    x1=sinusoid()(x1)
    
    x1=Concatenate(axis=-1)([GetReal()(x),GetReal()(x1),GetImag()(x),GetImag()(x1)])
    
    x2=ComplexConv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex', init_criterion='he', bias_initializer = 'zeros')(x1)
    x2=ComplexBatchNormalization()(x2)
    x2=sinusoid()(x2)

    x2=Concatenate(axis=-1)([GetReal()(x1),GetReal()(x2),GetImag()(x1),GetImag()(x2)])
        
    x3=ComplexConv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex', init_criterion='he', bias_initializer = 'zeros')(x2)
    x3=ComplexBatchNormalization()(x3)
    x3=sinusoid()(x3)

    x3=Concatenate(axis=-1)([GetReal()(x2),GetReal()(x3),GetImag()(x2),GetImag()(x3)])
    
    x4=ComplexConv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex', init_criterion='he', bias_initializer = 'zeros')(x3)
    x4=ComplexBatchNormalization()(x4)
    x4=sinusoid()(x4)

    x4=Concatenate(axis=-1)([GetReal()(x3),GetReal()(x4),GetImag()(x3),GetImag()(x4)])
    
    x5=ComplexConv2D(filters=fil,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex', init_criterion='he', bias_initializer = 'zeros')(x4)
    x5=Lambda(lambda x:x*beta)(x5)
    xout=Add()([x5,x])
    
    return xout

def resresden(x,fil,gr,betad,betar,gamma_init,trainable):
    x1=resden(x,fil,gr,betad,gamma_init,trainable)
    x2=resden(x1,fil,gr,betad,gamma_init,trainable)
    x3=resden(x2,fil,gr,betad,gamma_init,trainable)
    x3=Lambda(lambda x:x*betar)(x3)
    xout=Add()([x3,x])
    
    return xout

def generator(inp_shape, trainable = True): #training will at least have to be given - required for inference mode.
   gamma_init = tf.random_normal_initializer(1., 0.02)

   fd=32
   gr=8
   nb=4
   betad=0.2
   betar=0.2

   inp_real_imag = Input(inp_shape)
   pool_8to7 = AveragePooling2D(pool_size = (2,2), padding = 'same')(inp_real_imag)
   pool_8to6 = AveragePooling2D(pool_size = (2,2), padding = 'same')(pool_8to7)
   pool_8to5 = AveragePooling2D(pool_size = (2,2), padding = 'same')(pool_8to6)

   lay_128dn = ComplexConv2D(32, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(inp_real_imag)
   
   lay_128dn = sinusoid()(lay_128dn)
   pool_7to6 = AveragePooling2D(pool_size = (2,2), padding = 'same')(lay_128dn)
   pool_7to5 = AveragePooling2D(pool_size = (2,2), padding = 'same')(pool_7to6)
   pool_7to4 = AveragePooling2D(pool_size = (2,2), padding = 'same')(pool_7to5)

   lay_64dn = Concatenate(axis=-1)([GetReal()(pool_8to7), GetReal()(lay_128dn),GetImag()(pool_8to7), GetImag()(lay_128dn)])
    
   lay_64dn = ComplexConv2D(32, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_64dn)
   lay_64dn = ComplexBatchNormalization()(lay_64dn)
   lay_64dn = sinusoid()(lay_64dn)
   pool_6to5 = AveragePooling2D(pool_size = (2,2), padding = 'same')(lay_64dn)
   pool_6to4 = AveragePooling2D(pool_size = (2,2), padding = 'same')(pool_6to5)
   pool_6to3 = AveragePooling2D(pool_size = (2,2), padding = 'same')(pool_6to4)
   lay_32dn = Concatenate(axis=-1)([GetReal()(pool_8to6), GetReal()(pool_7to6), GetReal()(lay_64dn),GetImag()(pool_8to6), GetImag()(pool_7to6), GetImag()(lay_64dn)])
    
   lay_32dn = ComplexConv2D(32, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_32dn)
   lay_32dn = ComplexBatchNormalization()(lay_32dn)
   lay_32dn = sinusoid()(lay_32dn)
   pool_5to4 = AveragePooling2D(pool_size = (2,2), padding = 'same')(lay_32dn)
   pool_5to3 = AveragePooling2D(pool_size = (2,2), padding = 'same')(pool_5to4)
   lay_16dn = Concatenate(axis=-1)([GetReal()(pool_8to5), GetReal()(pool_7to5), GetReal()(pool_6to5), GetReal()(lay_32dn),GetImag()(pool_8to5), GetImag()(pool_7to5), GetImag()(pool_6to5), GetImag()(lay_32dn)])
    
   lay_16dn = ComplexConv2D(32, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_16dn)
   lay_16dn = ComplexBatchNormalization()(lay_16dn)
   lay_16dn = sinusoid()(lay_16dn)  #16x16 
   pool_4to3 = AveragePooling2D(pool_size = (2,2), padding = 'same')(lay_16dn)
   lay_8dn = Concatenate(axis=-1)([GetReal()(pool_7to4), GetReal()(pool_6to4), GetReal()(pool_5to4), GetReal()(lay_16dn), GetImag()(pool_7to4), GetImag()(pool_6to4), GetImag()(pool_5to4), GetImag()(lay_16dn)])
    
   lay_8dn = ComplexConv2D(32, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_8dn)
   lay_8dn = ComplexBatchNormalization()(lay_8dn)
   lay_8dn = sinusoid()(lay_8dn) #8x8
   xc1 = Concatenate(axis=-1)([GetReal()(pool_6to3), GetReal()(pool_5to3), GetReal()(pool_4to3), GetReal()(lay_8dn),GetImag()(pool_6to3), GetImag()(pool_5to3), GetImag()(pool_4to3), GetImag()(lay_8dn)])

   xc1=ComplexConv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(xc1) #8x8
   xrrd=xc1
   for m in range(nb):
     xrrd=resresden(xrrd,fd,gr,betad,betar,gamma_init,trainable)

   xc2=ComplexConv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(xrrd)
   xc2=Add()([xc1,xc2])
   up_3to4 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(xc2)
   up_3to5 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_3to4)
   up_3to6 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_3to5)
  
   lay_16up=UpSampling2D()(xc2)
   lay_16up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_16up) # confirm size wuth code, my guess is they are increasing size by 2 in every spatial dimension.
   lay_16up = ComplexBatchNormalization()(lay_16up)
   lay_16up = sinusoid()(lay_16up) #16x16
   up_4to5 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(lay_16up)
   up_4to6 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_4to5)
   up_4to7 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_4to6)

   lay_32up = Concatenate(axis = -1)([GetReal()(lay_16up),GetReal()(up_3to4),GetReal()(lay_16dn),GetImag()(lay_16up),GetImag()(up_3to4),GetImag()(lay_16dn)])
   
   lay_32up=UpSampling2D()(lay_32up) 
   lay_32up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_32up) # confirm size wuth code, my guess is they are increasing size by 2 in every spatial dimension.
   lay_32up = ComplexBatchNormalization()(lay_32up)
   lay_32up = sinusoid()(lay_32up) #32x32
   up_5to6 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(lay_32up)
   up_5to7 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_5to6)
    
   lay_64up = Concatenate(axis = -1)([GetReal()(lay_32up),GetReal()(up_3to5),GetReal()(up_4to5),GetReal()(lay_32dn),GetImag()(lay_32up),GetImag()(up_3to5),GetImag()(up_4to5),GetImag()(lay_32dn)])
    
   lay_64up=UpSampling2D()(lay_64up)
   lay_64up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_64up) # confirm size wuth code, my guess is they are increasing size by 2 in every spatial dimension.
   lay_64up = ComplexBatchNormalization()(lay_64up)
   lay_64up = sinusoid()(lay_64up) #64x64
   up_6to7 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(lay_64up)
    
   lay_128up = Concatenate(axis = -1)([GetReal()(lay_64up),GetReal()(up_3to6),GetReal()(up_4to6),GetReal()(up_5to6),GetReal()(lay_64dn),GetImag()(lay_64up),GetImag()(up_3to6),GetImag()(up_4to6),GetImag()(up_5to6),GetImag()(lay_64dn)])
   
   lay_128up=UpSampling2D()(lay_128up)  
   lay_128up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_128up) # confirm size wuth code, my guess is they are increasing size by 2 in every spatial dimension.
   lay_128up = ComplexBatchNormalization()(lay_128up)
   lay_128up = sinusoid()(lay_128up) #128x128
    
   lay_256up = Concatenate(axis = -1)([GetReal()(lay_128up),GetReal()(up_4to7),GetReal()(up_5to7),GetReal()(up_6to7),GetReal()(lay_128dn),GetImag()(lay_128up),GetImag()(up_4to7),GetImag()(up_5to7),GetImag()(up_6to7),GetImag()(lay_128dn)])
     
   lay_256up=UpSampling2D()(lay_256up)
   lay_256up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_256up) # confirm size wuth code, my guess is they are increasing size by 2 in every spatial dimension.
   lay_256up = ComplexBatchNormalization()(lay_256up)
   lay_256up = sinusoid()(lay_256up) #256x256
   
   out1 =  ComplexConv2D(1, (1,1), strides = (1,1), activation = 'tanh', padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_256up)
   out1=Lambda(lambda x:(x+1)/2)(out1)
   out1=GetAbs()(out1)
   out=Lambda(lambda x:np.sqrt(2)*x-1)(out1)

   model = Model(inputs = inp_real_imag, outputs = out)
   #model.summary()
   return model


gen4 = generator(inp_shape = inp_shape, trainable = False)

#to infer after a run
f = open('/home/Co-VeGAN/covegan_a5_metrics.txt', 'x')
f = open('/home/Co-VeGAN/covegan_a5_metrics.txt', 'a')

for i in range(120):
   filename = '/home/Co-VeGAN/covegan_a5_gen_%04d.h5' % (i+1)   
   gen4.load_weights(filename)
   out4 = gen4.predict(data_gen)
   psnr, ssim = metrics(data, out4[:,:,:,0],2.0)
   f.write('psnr = %.5f, ssim = %.7f' %(psnr, ssim))
   f.write('\n')
   print(psnr, ssim)


#to infer a single model
'''
i=30
filename = '/home/Co-VeGAN/covegan_a5_gen_%04d.h5' % (i+1)   
gen4.load_weights(filename)
out4 = gen4.predict(data_gen)
psnr, ssim = metrics(data, out4[:,:,:,0],2.0)
print(psnr,ssim)
'''
