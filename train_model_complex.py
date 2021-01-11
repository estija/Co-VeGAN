from keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf
import pickle
from keras.models import Model, Input
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense
from keras.initializers import RandomUniform
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D
from keras.layers import Flatten, Add
from keras.layers import Concatenate, Activation, Layer
from keras.layers import LeakyReLU, BatchNormalization, Lambda, Multiply
from conv import ComplexConv2D
from bn import ComplexBatchNormalization
from utils import GetReal, GetImag, GetAbs
from keras import backend as K
import os
from wvt_loss import wt_diff
from tensorflow.python.ops import array_ops


class sinusoid(Layer):
    #PC-SoS activation, as mentioned in the paper
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


def accw(y_true, y_pred):
  y_pred=K.clip(y_pred, -1, 1)
  return K.mean(K.equal(y_true, K.round(y_pred)))

def mssim(y_true, y_pred):
  costs = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
  return costs

def wloss(y_true,y_predict):
    return -K.mean(y_true*y_predict)


def discriminator(inp_shape = (256,256,1), trainable = True): 
    
    gamma_init = tf.random_normal_initializer(1., 0.02) 
    
    inp = Input(shape = inp_shape)
    
    l7 = Conv2D(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp) 
    l7 = LeakyReLU(alpha=0.2)(l7)
    
    l7 = Conv2D(64*2, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
    l7 = LeakyReLU(alpha=0.2)(l7)
    
    l7 = Conv2D(64*4, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
    l7 = LeakyReLU(alpha=0.2)(l7)
    
    l7 = Conv2D(64*4, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
    l7 = LeakyReLU(alpha=0.2)(l7)
    
    l7 = Conv2D(64*8, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
    l7 = LeakyReLU(alpha=0.2)(l7)
    
    l7 = Conv2D(64*8, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
    l7 = LeakyReLU(alpha=0.2)(l7)
    
    l7 = Conv2D(64*8, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
    l7 = LeakyReLU(alpha=0.2)(l7)
    
    l7 = Conv2D(64*4, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
    l7 = LeakyReLU(alpha=0.2)(l7)
    
    l11 = Conv2D(64*2, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l11 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l11)
    l11 = LeakyReLU(alpha=0.2)(l11)
    
    l11 = Conv2D(64*2, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l11)
    l11 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l11)
    l11 = LeakyReLU(alpha=0.2)(l11)
    
    l11 = Conv2D(64*4, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l11)
    l11 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l11)
    l11 = LeakyReLU(alpha=0.2)(l11)

    l11 = Add()([l7,l11])
    l11 = LeakyReLU(alpha = 0.2)(l11)
    
    out=Conv2D(filters=1,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l11)
    
    model = Model(inputs = inp, outputs = out)
    
    return model

def resden(x,fil,gr,beta,gamma_init,trainable):
    x1=ComplexConv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex',init_criterion='he', bias_initializer = 'zeros')(x)
    x1=ComplexBatchNormalization()(x1)
    x1=sinusoid()(x1)

    x1=Concatenate(axis=-1)([GetReal()(x),GetReal()(x1),GetImag()(x),GetImag()(x1)])

    x2=ComplexConv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex', init_criterion='he', bias_initializer = 'zeros')(x1)
    x2=ComplexBatchNormalization()(x2)
    x2=sinusoid()(x2)

    x2=Concatenate(axis=-1)([GetReal()(x1),GetReal()(x2),GetImag()(x1),GetImag()(x2)])

    x1=ComplexConv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex', init_criterion='he', bias_initializer = 'zeros')(x2)
    x1=ComplexBatchNormalization()(x1)
    x1=sinusoid()(x1)

    x1=Concatenate(axis=-1)([GetReal()(x2),GetReal()(x1),GetImag()(x2),GetImag()(x1)])

    x2=ComplexConv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex', init_criterion='he', bias_initializer = 'zeros')(x1)
    x2=ComplexBatchNormalization()(x2)
    x2=sinusoid()(x2)

    x2=Concatenate(axis=-1)([GetReal()(x1),GetReal()(x2),GetImag()(x1),GetImag()(x2)])

    x1=ComplexConv2D(filters=fil,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer='complex', init_criterion='he', bias_initializer = 'zeros')(x2)
    x1=Lambda(lambda x:x*beta)(x1)
    xout=Add()([x1,x])

    return xout


def resresden(x,fil,gr,betad,betar,gamma_init,trainable):
    x1=resden(x,fil,gr,betad,gamma_init,trainable)
    x1=resden(x1,fil,gr,betad,gamma_init,trainable)
    x1=resden(x1,fil,gr,betad,gamma_init,trainable)
    x1=Lambda(lambda x:x*betar)(x1)
    xout=Add()([x1,x])

    return xout

def generator(inp_shape, trainable = True):
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
   lay_16dn = sinusoid()(lay_16dn)
   pool_4to3 = AveragePooling2D(pool_size = (2,2), padding = 'same')(lay_16dn)
   lay_8dn = Concatenate(axis=-1)([GetReal()(pool_7to4), GetReal()(pool_6to4), GetReal()(pool_5to4), GetReal()(lay_16dn), GetImag()(pool_7to4), GetImag()(pool_6to4), GetImag()(pool_5to4), GetImag()(lay_16dn)])

   lay_8dn = ComplexConv2D(32, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_8dn)
   lay_8dn = ComplexBatchNormalization()(lay_8dn)
   lay_8dn = sinusoid()(lay_8dn) #8x8
   xc1 = Concatenate(axis=-1)([GetReal()(pool_6to3), GetReal()(pool_5to3), GetReal()(pool_4to3), GetReal()(lay_8dn),GetImag()(pool_6to3), GetImag()(pool_5to3), GetImag()(pool_4to3), GetImag()(lay_8dn)])

   xc1=ComplexConv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(xc1)
   xrrd=xc1
   for m in range(nb):
     xrrd=resresden(xrrd,fd,gr,betad,betar,gamma_init,trainable)

   xc2=ComplexConv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(xrrd)
   xc2=Add()([xc1,xc2])
   up_3to4 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(xc2)
   up_3to5 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_3to4)
   up_3to6 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_3to5)

   lay_16up=UpSampling2D()(xc2)
   lay_16up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_16up)
   lay_16up = ComplexBatchNormalization()(lay_16up)
   lay_16up = sinusoid()(lay_16up) #16x16
   up_4to5 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(lay_16up)
   up_4to6 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_4to5)
   up_4to7 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_4to6)

   lay_32up = Concatenate(axis = -1)([GetReal()(lay_16up),GetReal()(up_3to4),GetReal()(lay_16dn),GetImag()(lay_16up),GetImag()(up_3to4),GetImag()(lay_16dn)])

   lay_32up=UpSampling2D()(lay_32up)
   lay_32up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_32up)
   lay_32up = ComplexBatchNormalization()(lay_32up)
   lay_32up = sinusoid()(lay_32up) #32x32
   up_5to6 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(lay_32up)
   up_5to7 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up_5to6)

   lay_64up = Concatenate(axis = -1)([GetReal()(lay_32up),GetReal()(up_3to5),GetReal()(up_4to5),GetReal()(lay_32dn),GetImag()(lay_32up),GetImag()(up_3to5),GetImag()(up_4to5),GetImag()(lay_32dn)])

   lay_64up=UpSampling2D()(lay_64up)
   lay_64up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_64up)
   lay_64up = ComplexBatchNormalization()(lay_64up)
   lay_64up = sinusoid()(lay_64up) #64x64
   up_6to7 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(lay_64up)

   lay_128up = Concatenate(axis = -1)([GetReal()(lay_64up),GetReal()(up_3to6),GetReal()(up_4to6),GetReal()(up_5to6),GetReal()(lay_64dn),GetImag()(lay_64up),GetImag()(up_3to6),GetImag()(up_4to6),GetImag()(up_5to6),GetImag()(lay_64dn)])

   lay_128up=UpSampling2D()(lay_128up)
   lay_128up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_128up)
   lay_128up = ComplexBatchNormalization()(lay_128up)
   lay_128up = sinusoid()(lay_128up) #128x128

   lay_256up = Concatenate(axis = -1)([GetReal()(lay_128up),GetReal()(up_4to7),GetReal()(up_5to7),GetReal()(up_6to7),GetReal()(lay_128dn),GetImag()(lay_128up),GetImag()(up_4to7),GetImag()(up_5to7),GetImag()(up_6to7),GetImag()(lay_128dn)])

   lay_256up=UpSampling2D()(lay_256up)
   lay_256up = ComplexConv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_256up)
   lay_256up = ComplexBatchNormalization()(lay_256up)
   lay_256up = sinusoid()(lay_256up) #256x256

   out1 =  ComplexConv2D(1, (1,1), strides = (1,1), activation = 'tanh', padding = 'same', use_bias = True, kernel_initializer = 'complex', init_criterion='he', bias_initializer = 'zeros')(lay_256up)
   out2 = GetAbs()(out1)
   model = Model(inputs = inp_real_imag, outputs = [out1,out2])
   #model.summary()
   return model


def define_gan_model(gen_model, dis_model, inp_shape):

    dis_model.trainable = False
    inp = Input(shape = inp_shape)

    out_g, out_g_abs = gen_model(inp)
    #out_g_abs = GetAbs()(out_g)
    out_g_r = GetReal()(out_g)
    out_g_i = GetImag()(out_g)
    out_dis = dis_model(out_g_abs)
    model = Model(inputs = inp, outputs = [out_dis, out_g_r, out_g_i, out_g_abs, out_g_abs])

    return model


def train(g_par, d_par, gan_model, dataset_gt, u_sampled_data, dataset_gt_abs, n_epochs, n_batch, n_critic, clip_val, n_patch, f):

    bat_per_epo = int(dataset_gt.shape[0]/n_batch)
    half_batch = int(n_batch/2)

    for i in range(n_epochs):
        if i==30:
            n_critic=3

        for j in range(bat_per_epo):

            # training the discriminator
            for k in range(n_critic):
                ix = np.random.randint(0, dataset_gt.shape[0], half_batch)

                X_real = dataset_gt_abs[ix]
                y_real = np.ones((half_batch,n_patch,n_patch,1))

                ix_1 =  np.random.randint(0, u_sampled_data.shape[0], half_batch)
                _, X_fake  = g_par.predict(u_sampled_data[ix_1])
                y_fake = -np.ones((half_batch,n_patch,n_patch,1))

                X, y = np.vstack((X_real, X_fake)), np.vstack((y_real,y_fake))
                d_loss, acc = d_par.train_on_batch(X,y)

                for l in d_par.layers:
                    weights=l.get_weights()
                    weights=[np.clip(w, -clip_val,clip_val) for w in weights]
                    l.set_weights(weights)

            # training the generator
            ix = np.random.randint(0, dataset_gt.shape[0], n_batch)
            X_r = dataset_gt[ix]
            X_abs = dataset_gt_abs[ix]
            X_gen_inp = u_sampled_data[ix]
            y_gan = np.ones((n_batch,n_patch,n_patch,1))

            g_loss = gan_model.train_on_batch([X_gen_inp], [y_gan, np.expand_dims(X_r[:,:,:,0],axis=-1), np.expand_dims(X_r[:,:,:,1],axis=-1),  X_abs, X_abs])
            f.write('>%d, %d/%d, d=%.3f, acc = %.3f,  w=%.3f, mae=%.3f,  mssim=%.3f, wvt=%.7f, g=%.3f' %(i+1, j+1, bat_per_epo, d_loss, acc, g_loss[1], (g_loss[2]+g_loss[3])/2, g_loss[4], g_loss[5], g_loss[0]))
            f.write('\n')
            print ('>%d, %d/%d, d=%.3f, acc = %.3f,  w=%.3f, mae=%.3f,  mssim=%.3f, wvt=%.7f, g=%.3f' %(i+1, j+1, bat_per_epo, d_loss, acc, g_loss[1], (g_loss[2]+g_loss[3])/2, g_loss[4], g_loss[5], g_loss[0]))

        #saving generator weights
        file_name = '/home/Co-VeGAN/covegan_a5_gen_%04d.h5' % (i+1)
        g_save = g_par.get_layer('model_3')
        g_save.save_weights(file_name)

        #to save discriminator weights, optimizer states
        if (i+1)%5==0:
          file_name = '/home/Co-VeGAN/covegan_a5_disc_%04d.h5' % (i+1)
          d_save = d_par.get_layer('model_1')
          d_save.save_weights(file_name)

          symbolic_weights = getattr(d_par.optimizer, 'weights')
          weight_values = K.batch_get_value(symbolic_weights)
          with open('/home/Co-VeGAN/covegan_a5_disc_opt_%02d.pkl' %(i+1), 'wb') as f2:
           pickle.dump(weight_values, f2)
           f2.close()

          symbolic_weights = getattr(gan_model.optimizer, 'weights')
          weight_values = K.batch_get_value(symbolic_weights)
          with open('/home/Co-VeGAN/covegan_a5_gan_opt_%02d.pkl' %(i+1), 'wb') as f2:
           pickle.dump(weight_values, f2)
           f2.close()

          del symbolic_weights
          del weight_values

    f.close()


#hyperparameters
n_epochs =120
n_batch = 8
n_critic = 3
clip_val = 0.05
in_shape_gen = (320,320,2)
in_shape_dis = (320,320,1)

#commands to load opt states, model weights can be uncommented to resume training, replace 30 by appropriate epoch number
d_model = discriminator (inp_shape = in_shape_dis, trainable = True)
#filename_dis = '/home/Co-VeGAN/covegan_a5_disc_0030.h5'
#d_model.load_weights(filename_dis)
d_par = multi_gpu_model(d_model, gpus=4, cpu_relocation = True) #for multi-gpu training
#d_par.layers[-2].set_weights(d_model.get_weights())
opt = Adam(lr = 0.0001, decay = 0.0015, beta_1 = 0.5)
d_par.compile(loss = wloss, optimizer = opt, metrics = [accw])
'''
d_par._make_train_function()
with open('/home/Co-VeGAN/covegan_a5_disc_opt_30.pkl', 'rb') as f3:
   weight_values = pickle.load(f3)
d_par.optimizer.set_weights(weight_values)
#K.set_value(d_par.optimizer.lr,K.get_value(d_par.optimizer.lr)/4) #to change base lr
'''
g_model = generator(inp_shape = in_shape_gen , trainable = True)
#filename_gen = '/home/Co-VeGAN/covegan_a5_gen_0030.h5'
#g_model.load_weights(filename_gen)
g_par = multi_gpu_model(g_model, gpus=4, cpu_relocation = True) #for multi-gpu training
g_par.summary()
#g_par.layers[-3].set_weights(g_model.get_weights())

gan_model = define_gan_model(g_par, d_par, in_shape_gen)
opt1 = Adam(lr = 0.0002, decay = 0.0015, beta_1 = 0.5)
gan_model.compile(loss = [wloss, 'mae', 'mae', mssim, wt_diff], optimizer = opt1, loss_weights = [0.01, 20.0, 20.0, 1.0, 10.0]) #loss weights for generator training
'''
gan_model._make_train_function()
with open('/home/Co-VeGAN/covegan_a5_gan_opt_30.pkl', 'rb') as f3:
   weight_values = pickle.load(f3)
gan_model.optimizer.set_weights(weight_values)
#K.set_value(gan_model.optimizer.lr,K.get_value(gan_model.optimizer.lr)/4) #to change base lr
'''
n_patch=d_model.output_shape[1]

data_path='/home/Co-VeGAN/training_gt_aug.pickle' #Ground truth
usam_path='/home/Co-VeGAN/training_usamp_1dg_a5_aug.pickle' #Zero-filled reconstructions

df=open(data_path,'rb')
uf=open(usam_path,'rb')

dataset_gt=pickle.load(df)
u_sampled_data=pickle.load(uf)

dataset_gt = np.expand_dims(dataset_gt, axis = -1)
u_sampled_data = np.expand_dims(u_sampled_data, axis = -1)

dataset_gt_real = dataset_gt.real
dataset_gt_imag = dataset_gt.imag
dataset_gt_abs = np.abs(dataset_gt)
dataset_gt_2c = np.concatenate((dataset_gt_real, dataset_gt_imag), axis = -1)

u_sampled_data_real = u_sampled_data.real
u_sampled_data_imag = u_sampled_data.imag

r_max = np.max(u_sampled_data_real)
i_max = np.max(u_sampled_data_imag)
r_min = np.min(u_sampled_data_real)
i_min = np.min(u_sampled_data_imag)

max_val = np.max([np.abs(r_max), np.abs(i_max), np.abs(r_min), np.abs(i_min)])
#print(max_val)

u_sampled_data_real = u_sampled_data_real/max_val
u_sampled_data_imag = u_sampled_data_imag/max_val

u_sampled_data_2c = np.concatenate((u_sampled_data_real, u_sampled_data_imag), axis = -1)

f = open('/home/Co-VeGAN/covegan_a5_log.txt', 'x')
f = open('/home/Co-VeGAN/covegan_a5_log.txt', 'a')

train(g_par, d_par, gan_model, dataset_gt_2c, u_sampled_data_2c, dataset_gt_abs, n_epochs, n_batch, n_critic, clip_val, n_patch, f)
