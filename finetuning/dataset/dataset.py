from torchvision import transforms
from PIL import Image
import torch
import os
from pycocotools import mask
import glob
import itertools
import json
from collections import defaultdict
import numpy as np


def get_bbox(rle_encoded_mask):
    bbox = rle_encoded_mask['bbox']

    return bbox

def get_area(rle_encoded_mask):
    area = rle_encoded_mask['area']

    return area

def get_binary_mask(rle_encoded_mask):
    coco_rle = rle_encoded_mask['segmentation']
    binary_mask = mask.decode(coco_rle)

    return binary_mask

def ids2img_names(ids, dataset_info):
    ids_to_names_dict = {}
    for name, info in dataset_info.items():
        if info.get('id') in ids:
            ids_to_names_dict[info['id']] = name
    return ids_to_names_dict

def resize_mask(image):
    image = Image.fromarray(image)
    longest_edge = 256
    # get new size
    w, h = image.size
    scale = longest_edge * 1.0 / max(h, w)
    new_h, new_w = h * scale, w * scale
    new_h = int(new_h + 0.5)
    new_w = int(new_w + 0.5)

    resized_image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return resized_image

def pad_mask(image):
    pad_height = 256 - image.height
    pad_width = 256 - image.width

    padding = ((0, pad_height), (0, pad_width))
    padded_image = np.pad(image, padding, mode="constant")
    return padded_image

def process_mask(image):
    resized_mask = resize_mask(image)
    padded_mask = pad_mask(resized_mask)
    return padded_mask

def resize_bbox(bbox, img_size):
    """
    Resize bounding box coordinates based on a scale factor.
    
    Parameters:
        bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
        scale_factor (float): Scale factor for resizing.
        
    Returns:
        tuple: Resized bounding box coordinates.
    """

    longest_edge = 256
    # get new size
    w, h = img_size
    scale = longest_edge * 1.0 / max(h, w)

    x_min, y_min = bbox[:2]
    x_max = x_min + bbox[2]
    y_max = y_min +  bbox[3]

    # Calculate new coordinates based on the scale factor
    new_x_min = int(x_min * scale)
    new_y_min = int(y_min * scale)
    new_x_max = int(x_max * scale)
    new_y_max = int(y_max * scale)

    return new_x_min, new_y_min, new_x_max, new_y_max


# Define dataset with masks
class MagPartDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, device, transform=None, split='train'):
        self.images_dir = images_dir
        self.mask_dir = masks_dir
        self.transform = transform
        self.device = device
        self.split = split
        self.dataset_info = self.get_dataset_info(self.images_dir)
        self.img_ids = [self.dataset_info[key]['id'] for key in self.dataset_info.keys() if self.dataset_info[key]['split'] == self.split]
        self.img_map = ids2img_names(self.img_ids, self.dataset_info)

        self.transformed_data = self.get_img_mask_bbox(self.img_ids)

    def __len__(self):
        return len(self.img_ids)
    
    def get_dataset_info(self, images_dir):
        file_name = os.path.join(images_dir, "dataset_info.json")

        with open(file_name, "r") as json_file:
            dataset_info = json.load(json_file)

        return dataset_info
    
    def get_img_mask_bbox(self, ids):

        transformed_data_all = {}
        for img_id in ids:
            transformed_data = {}
            img_name = self.img_map[img_id]
            img_name = os.path.dirname(img_name) + '/' + os.path.splitext(os.path.basename(img_name))[0]
            print(img_name)
            img_path = os.path.join(self.images_dir, img_name + ".tif")
            masks_path = os.path.join(self.mask_dir, img_name + ".json")

            image = np.array(Image.open(img_path).convert('RGB'))

            if self.transform:
                input_image = self.transform(image)
            else:
                input_image = torch.tensor(image)

            original_image_size = image.shape[-3:-1]
            input_size = tuple(input_image.shape[-2:])

            transformed_data['image'] = input_image[0]

            # Load the masks
            with open(masks_path, "r") as json_file:
                rle_encoded_masks = json.load(json_file)
            
            bbox_coords = []
            ground_truth_masks = []

            for rle_encoded_mask in rle_encoded_masks:
                area = get_area(rle_encoded_mask)
                ground_truth_mask = get_binary_mask(rle_encoded_mask)
                ground_truth_mask = process_mask(ground_truth_mask)
                bbox = get_bbox(rle_encoded_mask)

                # Transform the bounding boxes as well
                ground_truth_bbox = resize_bbox(bbox, original_image_size)

                ground_truth_masks.append(torch.tensor(ground_truth_mask))
                bbox_coords.append(torch.tensor(ground_truth_bbox))
            
            transformed_data['ground_truth_masks'] = ground_truth_masks
            transformed_data['bboxes'] = bbox_coords
            transformed_data['input_size'] = torch.tensor(input_size)
            transformed_data['original_image_size'] = torch.tensor(original_image_size)
        
            transformed_data_all[img_id] = transformed_data

        return transformed_data_all
                

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        return self.transformed_data[img_id]

