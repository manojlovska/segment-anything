import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask
from helper_functions import refine_masks
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_path", type=str, required=True, help="path to the image"
    )
    parser.add_argument(
        "--save", type=bool, default=True, help="set to True for saving the plot"
    )
    parser.add_argument(
        "--save_path", type=str, default="./results/masks_plot.png", help="path to save the plot"
    )

    return parser.parse_args()

def show_output(result_dict, axes=None, refine=False):
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
                mask_binary = mask.decode(val['segmentation'])
                mask_binary = refine_masks(mask_binary)
        else:
            mask_binary = mask.decode(val['segmentation'])
        
        img = np.ones((mask_binary.shape[0], mask_binary.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, mask_binary * 0.5)))

def main(args):
    image_path = args.image_path

    mask_path = os.path.join("./output", image_path.split("/")[-2], image_path.split("/")[-1].split(".")[0] + ".json")

    _, axes = plt.subplots(1, 3, figsize=(20, 8))

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(mask_path, 'r') as f:
        annotations = json.load(f)

    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', fontsize=20)
    axes[1].set_title('Original Masks', fontsize=20)
    axes[2].set_title('Refined Masks', fontsize=20)

    show_output(annotations, axes[1], refine=False)
    show_output(annotations, axes[2], refine=True)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save_path)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)