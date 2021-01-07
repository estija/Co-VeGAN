import numpy as np
from skimage import measure

def psnrc(true_tensor,test_tensor,val=1):
    psnrt=0
    for i in range(true_tensor.shape[0]):
      mse=np.mean((np.abs(true_tensor[i,:,:,:]-test_tensor[i,:,:,:]))**2)
      psnrt+=10*np.log10(val**2/mse)
    psnrt=psnrt/true_tensor.shape[0]
    return psnrt

def metrics(true_tensor, test_tensor,max_val=1):
    psnrt = 0
    ssimt = 0
    for i in range(true_tensor.shape[0]): 

         psnr = measure.compare_psnr(true_tensor[i,:,:], test_tensor[i,:,:], data_range = max_val)
         ssim = measure.compare_ssim(true_tensor[i,:,:], test_tensor[i,:,:], data_range = max_val)
         psnrt = psnrt+psnr
         ssimt = ssimt+ssim

    psnrt = psnrt/true_tensor.shape[0]
    ssimt = ssimt/true_tensor.shape[0]
    return psnrt, ssimt
