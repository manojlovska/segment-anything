from typing import Any
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch

class SAMTransforms:
    def __init__(self, sam_model, device=None):
        """ sam_model = [model_type, model_checkpoint]"""
        self.device = device
        self.sam_model = sam_model_registry[sam_model[0]](sam_model[1])
        self.sam_model.to(device=device)
    
    def __call__(self, image):
        transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # Additional preprocessing
        input_image = self.sam_model.preprocess(input_image_torch)

        return input_image


        