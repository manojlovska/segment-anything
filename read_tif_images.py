import os
from PIL import Image
import matplotlib.pyplot as plt
import fnmatch
import cv2

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


image_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/TEM BSHF-DBSA-210325/' # BSHF-DBSA-210325_0001.tif'

images = find_images(image_path)

_,axes = plt.subplots(2,6, figsize=(20,5))
for i, image in enumerate(images):
    mask_name = image.split('.')[0] + '-a.tif'
    img = cv2.imread(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_name)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    axes[0][i].imshow(img_rgb)
    axes[1][i].imshow(mask_rgb)




