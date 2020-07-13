# Pytorch-Adaptive-Instance-Normalization

A Pytorch implementation of the 2017 Huang et. al. paper "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" [https://arxiv.org/abs/1703.06868](https://arxiv.org/abs/1703.06868)
Written from scratch with essentially no reference to Xun Huangs implementation in lua/torch (can be found here: [https://github.com/xunhuang1995/AdaIN-style](https://github.com/xunhuang1995/AdaIN-style)) but I'm none the less incredbily greatful to Huang et. al. for writing such an outstandingly beautiful paper and making their method so clear and easy to implement!
![Architecture](./architecture.jpg)

## Requirements

To run this model please install the latest version of pytorch, torchvision and CUDA.

## Loading Pretrained Weights

I have made a set of pretrained weights availabe on google drive if you don't want to train the model yourself. You can find them here [https://drive.google.com/file/d/1094pChApSOA7qJZn68kEdNxKIwPWRdHn/view?usp=sharing](https://drive.google.com/file/d/1094pChApSOA7qJZn68kEdNxKIwPWRdHn/view?usp=sharing).
Once downloaded just place it into the root directory of the repo and you're good to go. 

## Usage

To use the model for style transfer use the command `python style.pt <path to content image> <path to style image>`. 
The styled image will be saves as `output.jpg` in the currect directory.

## Traning The Model

To train the model from scratch first download the datasets you want to use. The paper uses this [https://www.kaggle.com/c/painter-by-numbers/data](https://www.kaggle.com/c/painter-by-numbers/data) Kaggle dataset of Wiki Art images as its soure for style images and the MS-COCO common objecs in context dataset [https://cocodataset.org/](https://cocodataset.org/) for its content images. After you've downloaded the datasets (or a subset of them as they are both pretty large, 10s of GB) place the style images in the `train/style` directory and the content images in the `train/content` directory.

To actully train the model just run `python -i train.py` which will start training and output previews of it's progress into the `tmp` directory every few interations.
Every epoch the model will be saved to a file called `adain_model`.

## To Do
* Add automatic gpu/cpu selection
* Add explanatory text to loss printout
* Implement Bias correction on moving average loss
* Update default hyperparameters to match that of Huang
* Train the model for longer and upload better pretrained weights
* Add command line options for hyperparameters
* Make `requirements.txt` file
* Add more advanced runtime style interpolation and masking features from the paper
* Add some examples to this readme
