import os
import argparse
import torch
from tqdm import tqdm
import cv2
import json
from typing import Any, Dict, List
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from helper_functions import find_images


def parse_args():
    parser = argparse.ArgumentParser()
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
    default="./models/sam_vit_b_01ec64.pth",
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
    )

    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
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
        "--convert-to-rle",
        action="store_true",
        help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
        ),
    )

    parser.add_argument(
        "--img-extension",
        type=str,
        default="tif",
        help="image extensions(default is tif)"
    )

    sam_settings = parser.add_argument_group("SAM Settings")

    sam_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    sam_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    sam_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.9,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    sam_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    sam_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    sam_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    sam_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    sam_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    sam_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    sam_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    sam_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )
    return parser.parse_args()

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
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
    
def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    kwargs = get_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **kwargs)

    if not os.path.isdir(args.img_path):
        images = [args.img_path]
    else:
        images = find_images(args.img_path, args.img_extension)

    os.makedirs(args.output, exist_ok=True)

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