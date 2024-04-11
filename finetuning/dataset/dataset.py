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

        self.transformed_data, self.ground_truth_masks, self.bbox_coords = self.get_img_mask_bbox(self.img_ids)

        # self.images = sorted([file for file in glob.glob(os.path.join(self.images_dir, '**', '*.tif'), recursive=True) 
        #                if not (file.endswith('-a.tif') or file.endswith('-1.tif'))])
        # self.gt_masks = self.get_masks_ann(self.masks_dir)


    def __len__(self):
        return len(self.img_ids)
    
    # def get_masks_ann(self, masks_path):
    #     "For given masks path, returns a list of lists with the paths of the annotations"

    #     masks_annotations = [list(g) for _, g in itertools.groupby(sorted(glob.glob(os.path.join(masks_path, '**', '*.json'), recursive=True)), os.path.dirname)]

    #     return masks_annotations
    
    def get_dataset_info(self, images_dir):
        file_name = os.path.join(images_dir, "dataset_info.json")

        with open(file_name, "r") as json_file:
            dataset_info = json.load(json_file)

        return dataset_info
    
    def get_img_mask_bbox(self, ids):
        bbox_coords = defaultdict(list)
        ground_truth_masks = defaultdict(list)
        transformed_data = defaultdict(dict)

        for img_id in ids:
            img_name = self.img_map[img_id]
            img_name = os.path.dirname(img_name) + '/' + os.path.splitext(os.path.basename(img_name))[0]
            print(img_name)
            img_path = os.path.join(self.images_dir, img_name + ".tif")
            masks_path = os.path.join(self.mask_dir, img_name + ".json")

            image = np.array(Image.open(img_path).convert('RGB'))

            if self.transform:
                input_image = self.transform(image)

            original_image_size = image.shape[-2:] # image.size[::-1]
            input_size = tuple(input_image.shape[-3:-1])

            transformed_data[img_id]['image'] = input_image[0]
            transformed_data[img_id]['input_size'] = input_size
            transformed_data[img_id]['original_image_size'] = original_image_size

            # Load the masks
            # file_name = os.path.join(masks_path, img_name.split('.')[0] + ".json")

            with open(masks_path, "r") as json_file:
                rle_encoded_masks = json.load(json_file)
            
            for rle_encoded_mask in rle_encoded_masks:
                area = get_area(rle_encoded_mask)
                ground_truth_mask = get_binary_mask(rle_encoded_mask)
                ground_truth_mask = process_mask(ground_truth_mask)
                bbox = get_bbox(rle_encoded_mask)

                ground_truth_masks[img_id].append(ground_truth_mask)
                bbox_coords[img_id].append(bbox)

        return transformed_data, ground_truth_masks, bbox_coords
                

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        return self.transformed_data[img_id], self.ground_truth_masks[img_id], self.bbox_coords[img_id]


# # Define transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)), # Resize to the size a model expects
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization values for pre-trained PyTorch models
# ])