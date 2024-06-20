import os
from finetuning.dataset import MagPartDataset
from finetuning.dataset import SAMTransforms
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
from finetuning.experiments.fine_tuning_exp import SAMExp


# Convert binary mask to color
def mask_to_color(mask):
    return cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, bbox):
    
    x_min, y_min = bbox[0], bbox[1]
    x_max, y_max = bbox[2], bbox[3]
    
    color = (255, 255, 255)
    
    cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)

# Visualize the dataset
def plot_examples(image, mask, bbox):

    fig, axes = plt.subplots(1,2)

    axes[0].imshow(image)
    axes[1].imshow(mask)

    # Create a Rectangle patch 
    rect = patches.Rectangle((bbox[0], bbox[1]), abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1]), linewidth=1, 
                            edgecolor='r', facecolor="none") 
    
    # Add the patch to the Axes 
    axes[1].add_patch(rect) 

    axes[0].grid()
    axes[1].grid()
    axes[0].set_title('Image')
    axes[1].set_title('Mask')
    plt.savefig('./tests/figures/example_dataset.png')


print("Start")

images_dir = "/home/matejm/anastasija/Materials-Science/segment-anything/data/Porazdelitev celcev original"
masks_dir = "/home/matejm/anastasija/Materials-Science/segment-anything/output"

# model = ["vit_b", "/home/matejm/anastasija/Materials-Science/segment-anything/models/sam_vit_b_01ec64.pth"]
# device = torch.device("cuda:0")

# transforms = SAMTransforms(sam_model=model, device=device)

# train_dataset = MagPartDataset(images_dir=images_dir, masks_dir=masks_dir, device=device, transform=transforms, split='train')


# image = train_dataset[0][0]["image"].cpu()
# # Convert the tensor to a numpy array
# numpy_image = image.numpy()

# # Convert the numpy array to a cv2 image
# cv2_image = np.transpose(numpy_image, (1, 2, 0))
# cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

# mask = train_dataset[0][1][1]
# bbox = train_dataset[0][2][1]

# plot_examples(cv2_image, mask, bbox)

#########################################################################################################################################################

sam_exp = SAMExp()
train_dataset = sam_exp.get_train_dataset()





import pdb
pdb.set_trace()