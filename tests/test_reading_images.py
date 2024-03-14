import os
import argparse
import torch
from PIL import Image
import glob
import os
import fnmatch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import json

def find_images(images_path):
    # List to store the paths of the PNG images
    tif_files = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(images_path):
        # Filter out PNG files that do not end with '-a.png'
        for filename in fnmatch.filter(filenames, '*.tif'):
            if not (filename.endswith('-a.tif') or filename.endswith('-1.tif')):
                tif_files.append(os.path.join(dirpath, filename))

    return tif_files

images_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/Sc0,5M 21.8.2018, 30kx, 4008x2672px'