# Co-VeGAN
This is the official implementation code for **[Co-VeGAN: Complex-Valued Generative Adversarial Network for Compressive Sensing MR Image Reconstruction](https://arxiv.org/abs/2002.10523)** by *Bhavya Vasudeva**, *Puneesh Deora**, *Saumik Bhattacharya*, *Pyari Mohan Pradhan* (*equal contribution).

## Pre-requisites
The code was written with Python 3.6.8 with the following dependencies:
* cuda release 9.0, V9.0.176
* tensorflow 1.12.0
* keras 2.2.4
* numpy 1.16.4
* scikit-image 0.15.0
* matplotlib 3.1.0
* nibabel 2.4.1
* cuDNN 7.4.1

This code has been tested in Ubuntu 16.04.6 LTS with 4 NVIDIA GeForce GTX 1080 Ti GPUs (each with 11 GB RAM).

## How to Use
### Preparing data
1. Downloading the dataset:

&nbsp; &nbsp; &nbsp; **MICCAI 2013 dataset:** 
* The MICCAI 2013 grand challenge dataset can be downloaded from this [webpage](https://my.vanderbilt.edu/masi/workshops/). It is required to fill a google form and register be able to download the data.

&nbsp;&nbsp;&bull; Download and save the 'training-training' and training-testing' folders, which contain the training and testing data, respectively, into the repository folder.

&nbsp; &nbsp; **MRNet dataset:** 
* The MRNet dataset can be downloading from this [webpage](https://stanfordmlgroup.github.io/competitions/mrnet/). It also requires to register by filling the form at the end of the page to be able to download the data.
* Download and save the 'train' and 'valid' folders, which contain the training and testing data, respectively, into the repository folder.

2. Run the following command to create the GT dataset:
```
python dataset_load.py
```
3. Run the following command to create the undersampled dataset:
```
python usamp_data.py
```
4. These files would create the training data using MICCAI 2013 dataset. For MRNet dataset, or for testing data, please read the comments in the files to make the necessary changes.
5. The `masks` folder contains the undersampling masks used in this work. The path for the mask can be modified in `usamp_data.py`, as required.

### Training
1. Move the files in `complexnn` folder to the repository folder.
2. Run the following command to train the model, after checking the names of paths:
```
python train_model.py
```

### Testing
#### Testing the trained model:
1. Run the following command to test the model, after checking the names of paths:
```
python test_model.py
```

#### Testing the pre-trained model:
1. The pre-trained generator weights for various undersampling patterns are available at: 
<p align="center">
<a href="https://drive.google.com/open?id=1WQ92TiBHJXplwwVDZ9jpY-lSBtvV9G6d">30% 1D-G</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
<a href="https://drive.google.com/open?id=1u5YC1zJDIk__RDCKrRppHfRXQSiKeupY">30% Radial</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
<a href="https://drive.google.com/open?id=1zAxyxs9bpag4iCV2jk4P71RrhO8ry8BS">30% Spiral</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
<a href="https://drive.google.com/open?id=1wXC322wti8eucKz9J39wZ2nRrjDezb_f">20% 1D-G</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
<a href="https://drive.google.com/open?id=1G60xAEr8na4AbPRtcRAtg6J--Re0j8-s">10% 1D-G</a>
</p>

Download the required weights in the repository folder. They can used to obtain the results as provided in the paper.
2. Run the following command, after changing the names of paths:
```
python test_model.py
```

## Citation
If you find our research useful, please cite our work.
```
@article{vasudeva2020covegan,
    title={Co-VeGAN: Complex-Valued Generative Adversarial Network for Compressive Sensing MR Image Reconstruction},
    author={B. Vasudeva and P. Deora and S. Bhattacharya and P. M. Pradhan},
    journal={ArXiv},
    year={2020},
    volume={abs/2002.10523}
}
```

## License
```
   Copyright 2020 Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
