# Co-VeGAN

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
1. The MICCAI 2013 grand challenge dataset can be downloaded from this [webpage](https://my.vanderbilt.edu/masi/workshops/). It is required to fill a google form and register be able to download the data.
2. Download and save the training and testing data in training-training and training-testing folders, respectively, into the repository folder.
3. Run 'python dataset_load.py' to create the GT dataset.
4. Run 'python usamp_data.py' to create the undersampled dataset. 
5. The 'masks' folder contains the undersampling masks used in this work. The path for the mask can be modified in the aformentioned file, as required.

### Training
1. Move the files in "complexnn" folder to the repository folder.
2. Run 'python train_model.py' to train the model, after checking the names of paths.

### Testing
#### Testing the trained model:
1. Run 'python test_model.py' to test the model, after checking the names of paths.

#### Testing the pre-trained model:
1. The pre-trained generator weights are available at: 
* 30% 1D-G undersampling
* 30% Radial undersampling
* 30% Spiral undersampling
* 20% 1D-G undersampling
* 10% 1D-G undersampling

Download the required weights in the repository folder. They can used to obtain the results as provided in the paper.
2. Run 'python test_model.py', after changing the names of paths.
