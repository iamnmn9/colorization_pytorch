# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:58:12 2021

@author: Naman Pundir
"""

import os

import glob
import shutil

os.makedirs('images/train/', exist_ok=True)  #3862 images
os.makedirs('images/val/', exist_ok=True)   #420 images


count = 1

train_dir = "images/train"
val_dir = "images/val"
src_dir = "landscape_images"

for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
	if count<421:
		shutil.copy(jpgfile, val_dir)
	else:
		shutil.copy(jpgfile, train_dir)
	count = count + 1
	   
	   