# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:22:19 2021

@author: jpeeples
"""
import numpy as np

def decode_segmap(image, nc=12):
   
    label_colors = np.array([(86, 115, 181), (132, 167, 77),(77, 77, 77),
                               (141, 202, 207),(211, 150, 202),(209, 205, 75),
                               (255, 196, 52), (51, 69, 83), (145, 58, 219),
                               (58, 69, 219), (80, 139, 48),(164, 38, 41)])
      
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb
