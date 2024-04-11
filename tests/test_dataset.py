import os
from finetuning.dataset import MagPartDataset
from finetuning.dataset import SAMTransforms
import torch

print("Start")

images_dir = "/work/anastasija/Materials-Science/segment-anything/data/Porazdelitev celcev original"
masks_dir = "/work/anastasija/Materials-Science/segment-anything/output_new"

model = ["vit_b", "/work/anastasija/Materials-Science/segment-anything/models/sam_vit_b_01ec64.pth"]
device = torch.device("cuda:0")

transforms = SAMTransforms(sam_model=model, device=device)

train_dataset = MagPartDataset(images_dir=images_dir, masks_dir=masks_dir, device=device, transform=transforms, split='train')

import pdb
pdb.set_trace()