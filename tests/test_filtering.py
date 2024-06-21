import cv2
import numpy as np
import json
from pycocotools import mask
import matplotlib.pyplot as plt


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

with open("/home/anastasija/Documents/IJS/E8/Magnetic-Particles/code/segment-anything/output/BSHF-DBSA-210325/BSHF-DBSA-210325_0001.json", 'r') as f:
    annotations = json.load(f)

annotation = annotations[2]

mask_binary = mask.decode(annotation["segmentation"])
mask_refined = refine_masks(mask_binary)

img = np.ones((mask_refined.shape[0], mask_refined.shape[1], 3))
color_mask = np.random.random((1, 3)).tolist()[0]

for i in range(3):
    img[:,:,i] = color_mask[i]

# Overlay the mask on the image with transparency
plt.imshow(np.dstack((img, mask_refined * 0.5)))
plt.axis('off')  # Turn off the axis
plt.show()


