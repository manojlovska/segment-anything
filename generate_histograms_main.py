import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from helper_functions import * 


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--masks-path", type=str, default="./output", help="path to SAM outputs folder"
    )
    parser.add_argument(
        "--images-path", type=str, required=True, help="path to the experimental analyses images"
    )
    parser.add_argument(
        "--save-hist", action="store_true", help="set to True for saving the histograms plot"
    )
    parser.add_argument(
        "--save-path", type=str, default="./results", help="path to save the histograms plot"
    )
    parser.add_argument(
        "--t-min", type=int, default=10, help="min diameter value(in nm) for the particle to be included in histogram analysis (default is 10)"
    )
    parser.add_argument(
        "--t-max", type=int, default=150, help="max diameter value(in nm) for the particle to be included in histogram analysis (default is 150)"
    )
    parser.add_argument(
        "--filter-edge", action="store_true", help="set for filtering edge particles"
    )
    parser.add_argument(
        "--ratio", type=float, required=True, help="ratio for converting area from px to nm"
    )
    return parser.parse_args()

def main(args):
    # Check if paths exist
    if not os.path.exists(args.masks_path):
        raise FileNotFoundError(f"Masks path '{args.masks_path}' does not exist.")
    if not os.path.exists(args.images_path):
        raise FileNotFoundError(f"Images path '{args.images_path}' does not exist.")
    if args.save_hist and not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # Initialize json_files
    json_files = {}

    # Metadata for generated masks
    if check_image_extension(args.images_path):
        dir_name = os.path.basename(os.path.dirname(args.masks_path))
        json_files[dir_name] = [args.masks_path]
    else:
        json_files = find_files(args.masks_path, '.json')
    
    if not json_files:
        raise FileNotFoundError("No JSON files found in the masks path.")

    # Thresholds
    t_min = args.t_min
    t_max = args.t_max

    # Determine the number of plots
    num_files = len(json_files)
    if num_files == 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(num_files // 5, 5, figsize=(40, 20))
        axes = axes.flatten()

    for i, k in enumerate(tqdm(list(json_files.keys()), desc="Processing experiments")):
        try:
            # Images inside the specific experiment analysis
            images_paths = json_files[k]
            images = [os.path.splitext(os.path.basename(path))[0] for path in images_paths]

            print(f'Experiment analysis: {k} \nImages: {images}')

            # Find the detected particles
            diams_nm_predicted = []
            for annotation, image_name in zip(tqdm(images_paths, desc="Processing images"), images):
                decoded_annotation = decode_annotation(annotation)
                if args.filter_edge:
                    masks_refined = [refine_masks(binary_mask) for binary_mask in decoded_annotation["binary_masks"]]
                    indices = [mask_refined.any() for mask_refined in masks_refined]
                else:
                    indices = [True for _ in range(len(decoded_annotation["binary_masks"]))]

                if "areas" not in decoded_annotation or not decoded_annotation["areas"]:
                    raise ValueError(f"No valid areas found in the annotation file {annotation}.")

                areas = decoded_annotation["areas"]
                areas = [area for idx, area in zip(indices, areas) if idx]

                diams_px = [calculate_diam(area) for area in areas]
                if image_name.startswith('BSHF') or image_name.startswith('TEM'):
                    diams_px = [diam_px * 3 for diam_px in diams_px]

                diams_nm = [args.ratio * diam_px for diam_px in diams_px]
                diams_nm = [diam_nm for diam_nm in diams_nm if t_min <= diam_nm <= t_max]
                diams_nm_predicted.extend(diams_nm)

            # Generate histograms
            histogram_predicted = generate_diam_nm_histogram(diams_nm_predicted, bins=[0.01, 10, 20.01, 30, 40.01, 50, 60.01, 70, 80.01, 90, 100.01, 110, 120.01, 130, 140.01, 150])
            weights_pred = histogram_predicted[0] / np.sum(histogram_predicted[0])

            # Save the predicted diameters to csv file
            predicted_diameters_df = pd.DataFrame(diams_nm_predicted, columns=["Diameter (nm)"])
            predicted_diameters_df.to_csv(os.path.join(args.save_path, f"{k}_predicted_diameters.csv"), index=False)

            # Plot histogram
            ax = axes[i]
            ax.hist(histogram_predicted[1][:-1], bins=histogram_predicted[1], weights=weights_pred, alpha=0.5, align='mid', color='red')
            ax.set_title(f'{k}')
            ax.legend([f'{k} - MASKS'], loc='upper right')
            ax.set_xlabel('Diameter (nm)')
            ax.set_ylabel('Relative number of particles')
            ax.grid(True)

        except Exception as e:
            print(f"Error processing experiment {k}: {e}")
            continue

    plt.tight_layout()
    if args.save_hist:
        plt.savefig(os.path.join(args.save_path, "histograms.png"))
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)
