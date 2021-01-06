import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import cmath

dataset='miccai' #dataset to be used: miccai or mrnet or fastmri
mode='train' #train or test
save_path='/home/Co-VeGAN/'
mask_path='/home/Co-VeGAN/masks/mask_1dg_a5.pickle' #path for the required mask
train_path='/home/Co-VeGAN/training_gt_aug.pickle'
test_path='/home/Co-VeGAN/testing_gt.pickle'

def usam_data(data_tensor, mask, dataset):
    data =[] 
    for i in range(data_tensor.shape[0]):
        fourier = np.fft.fft2(data_tensor[i,:,:])
        if dataset=='miccai' or dataset=='mrnet':
          fourier  = np.fft.fftshift(fourier)
        subsam_fourier = np.multiply(fourier,mask) #undersampling in k-space
        uncen_fourier = np.fft.ifftshift(subsam_fourier)
        zro_image = np.fft.ifft2(uncen_fourier) #zero-filled reconstruction
        data.append(zro_image)        
    data = np.asarray(data)
    return data 


def usam_data_noise(data_tensor, mask, noise_ratio, dataset):
    fft_data=[]
    data =[]
    for i in range(data_tensor.shape[0]):
        fourier = np.fft.fft2(data_tensor[i,:,:])
        if dataset=='miccai' or dataset=='mrnet':
          fourier  = np.fft.fftshift(fourier)
        fft_data.append(fourier)
    fft_data=np.asarray(fft_data)
    fft_std=np.std(fft_data)
    nstd=(noise_ratio*fft_std)/np.sqrt(2)
    insh=(fft_data.shape[1],fft_data.shape[2])
    for k in range(fft_data.shape[0]):    
        fft_noise=fft_data[k,:,:]+np.random.normal(0,nstd,insh)+1j*np.random.normal(0,nstd,insh) #adding noise
        subsam_fourier = np.multiply(fft_noise,mask) #undersampling in k-space
        uncen_fourier = np.fft.ifftshift(subsam_fourier)
        zro_image = np.fft.ifft2(uncen_fourier) #zero-filled reconstruction
        data.append(zro_image) 
        
    data = np.asarray(data)
    return data 


maf=open(mask_path,'rb')
mask=pickle.load(maf)

if mode=='train':
  #creating undersampled training data
  trf=open(train_path,'rb')
  train_data=pickle.load(trf)
  
  if dataset=='miccai':
    train_data_new = usam_data(train_data[0:13461,:,:], mask, dataset)                  #no noise
    train_data_new2 = usam_data_noise(train_data[13461:13956,:,:], mask, 0.1, dataset)  #10% noise-overlapping
    train_data_new3 = usam_data_noise(train_data[13956:14451,:,:], mask, 0.2, dataset)  #20% noise-overlapping
    train_data_new4 = usam_data_noise(train_data[14451:17619,:,:,], mask, 0.1, dataset) #10% noise-nonoverlapping
    train_data_new5 = usam_data_noise(train_data[17619:20787,:,:], mask, 0.2, dataset)  #20% noise-nonoverlapping
  else:
    train_data_new = usam_data(train_data[0:8100,:,:], mask, dataset)                  #no noise
    train_data_new2 = usam_data_noise(train_data[8100:8400,:,:], mask, 0.1, dataset)  #10% noise-overlapping
    train_data_new3 = usam_data_noise(train_data[8400:8700,:,:], mask, 0.2, dataset)  #20% noise-overlapping
    train_data_new4 = usam_data_noise(train_data[8700:10600,:,:,], mask, 0.1, dataset) #10% noise-nonoverlapping
    train_data_new5 = usam_data_noise(train_data[10600:12500,:,:], mask, 0.2, dataset)  #20% noise-nonoverlapping
  
  stack1 = np.vstack((train_data_new,train_data_new2))
  stack2 = np.vstack((train_data_new3, train_data_new4))
  stack3 = np.vstack((stack2, train_data_new5))
  fstack = np.vstack((stack1, stack3))
  
  with open(os.path.join(save_path,'training_usamp_1dg_a5_aug.pickle'),'wb') as f:
    pickle.dump(fstack,f,protocol=4)
    
else:
  #creating undersampled testing data
  tef=open(test_path,'rb')
  test_data=pickle.load(tef)
  
  test_data_new = usam_data(test_data,mask,dataset) #for noise-free imgs
  #test_data_new=usam_data_noise(test_data,mask,0.1,dataset) #for imgs with 10% noise
  #test_data_new=usam_data_noise(test_data,mask,0.2,dataset) #for imgs with 20% noise
  
  with open(os.path.join(save_path,'testing_usamp_1dg_a5.pickle'),'wb') as f:
    pickle.dump(test_data_new,f,protocol=4)
