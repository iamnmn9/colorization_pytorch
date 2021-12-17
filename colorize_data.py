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

from torchvision.transforms.functional import resize

class ColorizeData(Dataset):
    def __init__(self,path):
        self.path1 = path
        
        # Initialize dataset, you may use a second dataset for validation if require
        # print('init m eagya')
        # print(len(path))
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __len__(self) -> int:
        # print("check2")
        # print(len(self.path1))
        return len(self.path1) # return Length of datasetreturn len(self)
        
        pass
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # print('chec3k')
        # print(index)
        # if torch.is_tensor(index):
        #   index = index.tolist()
        image = cv2.imread(self.path1[index])
        # print(image.shape) #3channel
        self.inputt = self.input_transform(image) #bnw
        # print("COLORIZEDTATAAA SHAPE")
        # print(self.inputt.shape)
        self.targett = self.target_transform(image) #colored
        # print('targt')
        # print(self.targett.shape)
        # print('check')
        return self.inputt, self.targett # Return the input tensor and output tensor for training
        pass