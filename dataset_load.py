import os
import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle

dataset='miccai' #dataset to be used: miccai or mrnet or fastmri
mode='train' #train or test
save_path='/home/Co-VeGAN/'

def load_a(path, num):
    f = os.listdir(path)
    a = len(f)
    data = []
    #use imgs with more than 10% non-zero values
    n_zero_ratio = 0.1
    #num is to reduce the number of files loaded
    for i in range(len(f)-num):
        img = os.path.join(path, f[i])
        img_l = nib.load(img)
        img_data = img_l.get_fdata()
        vol_max = np.max(img_data)
        img_data = img_data/vol_max*2
        for j in range(img_data.shape[2]): 
            if (float(np.count_nonzero(img_data[:,:,j]))/np.prod(img_data[:,:,j].shape))>=n_zero_ratio:
                img_data[:,:,j] = img_data[:,:,j]-1   
                img_data_ts = np.rot90(img_data[:,:,j])
                data.append(img_data_ts)
    data = np.asarray(data)
    return data

def load_b(path):
    f = os.listdir(path)
    data = []
    #use imgs with more than 10% non-zero values
    n_zero_ratio = 0.1
    for i in range(len(f)):
      img = os.path.join(path, f[i])
      data_new=np.load(img, allow_pickle =True )
      data_new=data_new.astype('float32')
      for j in range(data_new.shape[0]): 
        if (float(np.count_nonzero(data_new[j,:,:]))/np.prod(data_new[j,:,:].shape))>=n_zero_ratio:
          data_new[j,:,:] = data_new[j,:,:]/127.5-1.0    
          data.append(data_new[j,:,:])
    data = np.asarray(data)
    return data

def load_c(path,num,mode):
    fl = os.listdir(path)
    data = []
    #use imgs with more than 10% non-zero values
    n_zero_ratio = 0.1
    k = 0
    for i in range(num):
      if mode=='train':
        filename = os.path.join(path, fl[i])
      else:
        filename = os.path.join(path, fl[i+700]) 
      f = h5py.File(filename,'r')
      data_new = f['kspace']
      data_new = np.asarray(data_new)
      for j in range(data_new[0]):
        #ifft
        knee = np.fft.fftshift(np.fft.ifft2(data_new[j,:,:]))
        #crop 320x320 from centre
        knee = knee[data_new.shape[1]//2-160:data_new.shape[1]//2+160, data_new.shape[2]//2-160:data_new.shape[2]//2+160]
        data.append(knee)
      s = k
      vol = np.asarray(data)
      k = vol.shape[0]
      maxr = np.max(np.real(vol[s:k]))
      maxi = np.max(np.imag(vol[s:k]))
      minr = np.min(np.real(vol[s:k]))
      mini = np.min(np.imag(vol[s:k]))
      vol[s:k] = vol[s:k]/np.max([maxr,maxi,np.abs(minr),np.abs(mini)])
      data = list(vol)

    data = np.asarray(data)
    return data


def train_data_aug(train_gt,dataset):
    if dataset=='miccai':
        gt1=train_gt[0:13461,:,:]

        gt2a=train_gt[12471:12966,:,:]  #overlapping data
        gt2b=train_gt[12966:13461,:,:]  #overlapping data

        gt3a=train_gt[13461:16629,:,:]  #non-overlapping data
        gt3b=train_gt[16629:19797,:,:]  #non-overlapping data

    else:
        gt1=train_gt[0:8100,:,:]

        gt2a=train_gt[7500:7800,:,:]  #overlapping data
        gt2b=train_gt[7800:8100,:,:]  #overlapping data

        gt3a=train_gt[8100:10000,:,:]  #non-overlapping data
        gt3b=train_gt[10000:11900,:,:]  #non-overlapping data

    gt2=np.vstack((gt2a,gt2b))
    gt3=np.vstack((gt3a,gt3b))
    gt4 = np.vstack((gt2,gt3))

    gt_new=np.vstack((gt1,gt4))

    return gt_new


if mode=='train':
  if dataset=='miccai':
    train_path='/home/Co-VeGAN/training-training/warped-images'
    train_gt=load_a(train_path, 1090)
  elif dataset=='mrnet':
    train_path='/home/Co-VeGAN/train/coronal'
    train_gt=load_b(train_path)
  elif dataset=='fastmri':
    train_path='/home/Co-VeGAN/singlecoil_train'
    train_gt=load_c(train_path, 450, 'train')

  train_gt_aug=train_data_aug(train_gt,dataset) #created gt for augmented data

  with open(os.path.join(save_path,'training_gt_aug.pickle'),'wb') as f:
        pickle.dump(train_gt_aug,f,protocol=4)
else:
  if dataset=='miccai':
    test_path='/home/Co-VeGAN/training-testing/warped-images'
    test_data=load_a(test_path, 390)
  elif dataset=='mrnet':
    test_path='/home/Co-VeGAN/valid/coronal'
    test_data=load_b(test_path)
  elif dataset=='fastmri':
    test_path='/home/Co-VeGAN/singlecoil_train'
    test_data=load_c(test_path, 100, 'test')

  with open(os.path.join(save_path,'testing_gt.pickle'),'wb') as f:
       pickle.dump(test_data,f,protocol=4)
