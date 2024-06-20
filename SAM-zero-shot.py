import os
import argparse
import torch
from PIL import Image
import glob
import os
import fnmatch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import json
from typing import Any, Dict, List
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )
    parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )

    parser.add_argument(
        "--convert-to-rle",
        action="store_true",
        help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
        ),
    )
    parser.add_argument(
    "--pred_iou_thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    return parser.parse_args()


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

def find_jpg_images(images_path):
    # List to store the paths of the PNG images
    jpg_files = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(images_path):
        # Filter out PNG files that do not end with '-a.png'
        for filename in fnmatch.filter(filenames, '*.jpg'):
            if not (filename.endswith('-a.jpg') or filename.endswith('-1.jpg')):
                jpg_files.append(os.path.join(dirpath, filename))

    return jpg_files

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

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        # mask = refine_masks(mask)
        if mask.any():
            # filename = f"{i}.png"
            # cv2.imwrite(os.path.join(path, filename), mask * 255)
            mask_metadata = [
                str(i),
                str(mask_data["area"]),
                *[str(x) for x in mask_data["bbox"]],
                *[str(x) for x in mask_data["point_coords"][0]],
                str(mask_data["predicted_iou"]),
                str(mask_data["stability_score"]),
                *[str(x) for x in mask_data["crop_box"]],
            ]
            row = ",".join(mask_metadata)
            metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def get_kwargs(args):
    kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return kwargs
    

# def main(args):
#     # load model
#     sam = sam_model_registry["vit_b"](checkpoint="/work/anastasija/Materials-Science/segment-anything/models/sam_vit_b_01ec64.pth")
#     _ = sam.to(device=args.device)
#     mask_generator = SamAutomaticMaskGenerator(sam)
#     output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

#     images = find_images(args.img_path)

#     for image in tqdm(images):
#         input = cv2.imread(image)
#         if image is None:
#             print(f"Could not load '{image}' as an image, skipping...")
#             continue
#         input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

#         masks = mask_generator.generate(input)

#         base = os.path.basename(image)
#         base = os.path.splitext(base)[0]
#         save_base = os.path.join(args.output, image.split("/")[-2], base)

#         os.makedirs(save_base, exist_ok=True)
#         write_masks_to_folder(masks, save_base)

#         for item in masks:                                                                                                                                                                                                                                                                 
#             item['segmentation'] = item['segmentation'].tolist()

#         save_file = save_base + ".json"
#         with open(save_file, "w") as f:
#             json.dump(masks, f)
    
        
#         torch.cuda.empty_cache()
#     print("Done!")

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    # kwargs = get_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, pred_iou_thresh=args.pred_iou_thresh)

    # if not os.path.isdir(args.input):
    #     targets = [args.input]
    # else:
    #     targets = [
    #         f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
    #     ]
    #     targets = [os.path.join(args.input, f) for f in targets]

    images = find_jpg_images(args.img_path)

    os.makedirs(args.output, exist_ok=True)

    # import pdb
    # pdb.set_trace()

    for im in tqdm(images):
        print(f"Processing '{im}'...")
        image = cv2.imread(im)
        if image is None:
            print(f"Could not load '{im}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        base = os.path.basename(im)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, im.split("/")[-2], base)
        save_dir = os.path.join(args.output, im.split("/")[-2])

        # import pdb
        # pdb.set_trace()

        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=False)
            write_masks_to_folder(masks, save_base)
        else:
            os.makedirs(save_dir, exist_ok=True)
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)