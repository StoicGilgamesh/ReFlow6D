#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:38:41 2023

@author: stoic_gilgamesh
"""

import sys
import os
import subprocess
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage, signal
import math
import datetime
import copy
import transforms3d as tf3d
import time
import random
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import shift
​
​
if __name__ == '__main__':
​
    bg_path = "/hdd/test_matting/COCO_train2014_000000000009.jpg"
    rf_path = "/hdd/test_matting/000127_4_flow.png"
    rho_path = "/hdd/test_matting/000127_4_rho.png"
    mask_path = "/hdd/test_matting/000127_4_mask.png"
​
    bg = cv2.imread(bg_path, 1)
    rf = cv2.imread(rf_path, -1)
    rho = cv2.imread(rho_path, 1)
    mask = cv2.imread(mask_path, 0)
    
    y_shape, x_shape, _ = rf.shape
    bg = cv2.resize(bg[:480, :480, :], (x_shape, y_shape))
    x_grid = np.tile(np.linspace(0, x_shape-1, x_shape), (x_shape, 1)).astype(int)
    y_grid = np.tile(np.linspace(0, y_shape-1, y_shape), (y_shape, 1)).T.astype(int)
    
    #with open("/hdd/test_matting/000127_4_flow.flo", 'r') as f:
   # 	rf_bin = np.fromfile(f, dtype=np.int16)
    	
    #rf_bin = rf_bin[6:].reshape([512, 512, 2])
    #x_off = rf_bin[:, :, 1]
    #y_off = rf_bin[:, :, 0]
    
    #rf = cv2.GaussianBlur(rf,(5,5),2)
    #rf = cv2.medianBlur(rf,5)
    #cv2.imshow('rf', rf_bin[:, :, 1])
    #cv2.waitKey(0)
​
    # invert flowwithRho
    #rho = 255 - rho
    #rf = ((rf * 255) / rho) 
​
    #rf = cv2.cvtColor(rf, cv2.COLOR_BGR2HSV)
    rf = matplotlib.colors.rgb_to_hsv(rf/255)
​
    F_dir = rf[:, :, 0]
    F_mag = rf[:, :, 1]
    
    # invert flow_color[:,:,1] = F_mag / (F_mag.shape[0]*0.5)
    F_mag = F_mag * (y_shape * 0.5) # already pixel distance?
    
    # invert flow_color[:,:,0] = (F_dir+np.pi) / (2 * np.pi)
    F_dir = (F_dir * (2.0 * np.pi)) - np.pi
    
    # invert F_dir = np.arctan2(F_dy, F_dx)
    # https://stackoverflow.com/questions/11394706/inverse-of-math-atan2
    # local dx = len * cos(theta)
    # local dy = len * sin(theta)
    x_off = F_mag * np.cos(F_dir)
    y_off = F_mag * np.sin(F_dir)
    
    #x_off[mask >= 200] = 0
    #y_off[mask >= 200] = 0
    
    x_off += x_grid
    y_off += y_grid	
    
    corr = y_off + x_off
    #corr = np.multiply(y_off, x_shape)
    #corr = corr + x_off
    corr = corr.astype(np.int16)
    
    bg_flat = bg.reshape((y_shape * x_shape, 3))
    
    # refractive flow
    comp = bg_flat[corr.flatten(), :]
    comp = comp.reshape([y_shape, x_shape, 3])
    mask_rep = np.repeat((mask)[:, :, np.newaxis], axis=2, repeats=3)
    comp = np.where(mask_rep <= 200, comp, bg)
    
    #cv2.imshow('rf applied', comp)
    #cv2.waitKey(0)
    
    #attenuation
    #rho = cv2.medianBlur(rho,5)
    rho = (255 - rho) / 255
    rho[mask_rep >= 200] = 0 
    #print(rho.shape)
    #rho = np.repeat((rho)[:, :, np.newaxis], axis=2, repeats=3)
    comp = comp * (1 - rho) + rho*255
    #comp = (1 - rho) * 255 
    #print(np.min(comp), np.max(comp))
    
    cv2.imshow('attenuation applied', comp/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
​