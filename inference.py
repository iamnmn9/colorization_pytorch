# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 2:16:14 2021

@author: Naman Pundir
"""
from basic_model import Net
from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from PIL import Image
import io
import glob
from PIL import Image
from skimage import io
import os, os.path
import imageio
from torchvision.transforms import ToTensor
import numpy
import cv2
import matplotlib
from torchvision.utils import save_image


#######################IF CPU DEVICE###############################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net()
model.load_state_dict(torch.load('model.pth', map_location=device))

model.to(device)
#####################################################################################




#################### IF CUDA DEVICE ####################################
#model = Net()  # Initialize model
#model.load_state_dict(torch.load('model.pth'))  # Load pretrained parameters
###########################################################################


model.eval()  # Set to eval mode to change behavior of Dropout, BatchNorm

transform = T.Compose(    [T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ]) # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...

img = Image.open('black-white-lotus-flower-isolated-260nw-1816183571.jpg')  # Load image as PIL.Image
x = transform(img)  # Preprocess image
x = x.unsqueeze(0)  # Add batch dimension

output = model(x)  # Forward pass

output = output.squeeze(0)
print(output.shape)
print(output)
xx = output.detach()
save_image(xx,'ch.png')