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
from typing import Any, Dict, List
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

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

def im_clear_borders(thresh):
    kernel2 = np.ones((3,3), np.uint8)
    marker = thresh.copy()
    marker[1:-1,1:-1] = 0
    while True:
        tmp = marker.copy()
        marker = cv2.dilate(marker, kernel2)
        marker = cv2.min(thresh, marker)
        difference = cv2.absdiff(marker, tmp)
        if cv2.countNonZero(difference) == 0:
            break
    mask = cv2.bitwise_not(marker)
    out = cv2.bitwise_and(thresh, mask)
    return out

def refine_masks(mask):
    # https://stackoverflow.com/questions/65534370/remove-the-element-attached-to-the-image-border
    mask = im_clear_borders(mask.astype(np.uint8)).astype(bool)
    return mask

def show_output(result_dict,axes=None, refine=False):
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)

     # Plot for each segment area
    for val in sorted_result:
        if refine:
            if val['area'] > 250000.0 or val['area'] < 1000.0:
                continue
            else:
                mask = np.array(val['segmentation'])
                mask = refine_masks(mask)
        else:
            mask = np.array(val['segmentation'])

        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, mask*0.5)))

images_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/TEM BSHF-DBSA-210325/'
images = sorted(find_images(images_path))
masks_path = '/work/anastasija/Materials-Science/segment-anything/output/TEM BSHF-DBSA-210325/'

_,axes = plt.subplots(3,6, figsize=(20,10))
for i, image in enumerate(images):
    mask_name = image.split('.')[0] + '-a.tif'
    img = cv2.imread(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_name)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    json_file_path = os.path.join(masks_path, os.path.basename(image.split('.')[0]) + '.json')

    f = open(json_file_path)
 
    # returns JSON object as 
    # a dictionary
    json_dict = json.load(f)

    axes[0][i].imshow(img_rgb)
    axes[1][i].imshow(mask_rgb)

    show_output(json_dict, axes[2][i], refine=True)

# image_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/TEM BSHF-DBSA-210325/BSHF-DBSA-210325_0001.tif'
# gt_mask_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/TEM BSHF-DBSA-210325/BSHF-DBSA-210325_0001-a.tif'

# masks_path = '/work/anastasija/Materials-Science/segment-anything/output/TEM BSHF-DBSA-210325/'

# json_file_path = os.path.join(masks_path, 'BSHF-DBSA-210325_0001.json')
# # Opening JSON file
# f = open(json_file_path)
 
# # returns JSON object as 
# # a dictionary
# json_dict = json.load(f)

# image = cv2.imread(image_path)
# # Convert to RGB format
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Ground truth mask
# mask = cv2.imread(gt_mask_path)
# # Convert to RGB format
# mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# _,axes = plt.subplots(3,1, figsize=(16,16))
# axes[0].imshow(image_rgb)
# axes[1].imshow(mask_rgb)
# axes[0].set_title('Image')
# axes[1].set_title('Ground truth mask')
# axes[2].set_title('Generated mask')
# show_output(json_dict, axes[2])




