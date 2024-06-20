import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple
from tabulate import tabulate
from .base_exp import BaseClass

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LinearLR
import os
from torchvision import transforms

class SAMExp(BaseClass):
    """Basic class for any experiment."""
    def __init__(self):
        super().__init__()

        # Project name
        self.project_name = "SAM-fine-tunning"

        # number of workers
        self.data_num_workers = 1

        # device
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # pin memory
        self.pin_memory = True

        # directory of dataset images and masks
        self.images_dir = "/home/matejm/anastasija/Materials-Science/segment-anything/data/Porazdelitev celcev original"
        self.masks_dir = "/home/matejm/anastasija/Materials-Science/segment-anything/output"

        # SAM checkpoint path
        self.sam_path = "/home/matejm/anastasija/Materials-Science/segment-anything/models/sam_vit_b_01ec64.pth"

        # Output directory
        self.output_dir = "./SAM_outputs"

        # Choose SAM model type
        self.sam_type = "vit_b"

        # -------------- training config --------------- #
        # max training epochs
        self.max_epoch = 50
        self.max_lr = 0.01
        self.lr = 0.000001
        self.grad_clip = 0.1
        self.weight_decay = 1e-4

        # shuffle set to true for training
        self.shuffle_train = True

        # optimizer
        self.optimizer = torch.optim.Adam
        
        # learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CyclicLR

        # loss function -> Either Focal Loss or Dice Loss
        self.loss = "focal"
        self.loss_weight = 0.2
        self.size_average = True

        # if set to 1, user could see log every iteration
        self.print_interval = 10

        # eval period in epoch
        self.eval_interval = 1

        # save history checkpoint or not
        self.save_history_ckpt = True

        # -----------------  testing config ------------------ #
        # shuffle set to true for training
        self.shuffle_test = False
        self.return_predictions = True


    def get_model(self):
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry[self.sam_type](self.sam_path)

        return sam_model

    def get_train_dataset(self):
        from finetuning.dataset import SAMTransforms
        from finetuning.dataset import MagPartDataset

        transforms = SAMTransforms(sam_model=self.get_model(), device=self.device)

        return MagPartDataset(
            images_dir=self.images_dir, 
            masks_dir=self.masks_dir, 
            device=self.device, 
            transform=transforms, 
            split='train'
            )

    def get_train_loader(self, batch_size):
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self.get_train_dataset(), 
            batch_size, 
            shuffle=True,
            pin_memory=self.pin_memory
            )
        
    
    def get_eval_dataset(self):
        from finetuning.dataset import SAMTransforms
        from finetuning.dataset import MagPartDataset

        transforms = SAMTransforms(sam_model=self.get_model(), device=self.device)

        return MagPartDataset(
            images_dir=self.images_dir, 
            masks_dir=self.masks_dir, 
            device=self.device, 
            transform=transforms, 
            split='val'
            )
    
    def get_eval_loader(self, batch_size, device):
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self.get_eval_dataset(), 
            batch_size, 
            shuffle=self.shuffle_test, 
            num_workers=self.data_num_workers, 
            pin_memory=self.pin_memory)
    
    def get_loss_function(self, loss_name):
        if loss_name == "focal":
            from finetuning.losses import FocalLoss
            loss = FocalLoss(weight=self.loss_weight, size_average=self.size_average)
        else:
            from finetuning.losses import DiceLoss
            loss = DiceLoss(weight=self.loss_weight, size_average=self.size_average)
        
        return loss


    def get_trainer(self, args):
        from finetuning.train.trainer import Trainer

        trainer = Trainer(self, args)

        return trainer

    
    # def eval(self, model, val_loader, device):
    #     from plantdisease.utils.evaluators.evaluate import evaluate

    #     return evaluate(model, val_loader, device, return_predictions=self.return_predictions)
