import json
import numpy as np
from pycocotools import mask
import matplotlib.pyplot as plt

json_file = "output_new_data/5-5-11/dl1528 4008x2672 0,571621 12k.json"

with open(json_file, 'r') as f:
    annotations = json.load(f)

coco_rle = annotations[-1]['segmentation']

binary_mask = mask.decode(coco_rle)

# Visualize the binary mask
plt.imshow(binary_mask, cmap='gray')
plt.axis('off')
plt.title('Binary Mask')
plt.savefig("blabla.png")
plt.show()