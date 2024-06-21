import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from helper_functions import * # find_files, read_excel, decode_annotation, refine_masks, calculate_diam, generate_diam_nm_histogram, chi2_distance

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
    return parser.parse_args()

def main(args):
    print(f"filter edge: {args.filter_edge}")
    # Check if paths exist
    if not os.path.exists(args.masks_path):
        raise FileNotFoundError(f"Masks path '{args.masks_path}' does not exist.")
    if not os.path.exists(args.images_path):
        raise FileNotFoundError(f"Images path '{args.images_path}' does not exist.")
    if args.save_hist and not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # Metadata for generated masks
    json_files = find_files(args.masks_path, '.json')
    if not json_files:
        raise FileNotFoundError("No JSON files found in the masks path.")

    # Excel files with ground truths
    excel_files = find_files(args.images_path, '.xlsx')
    if not excel_files:
        raise FileNotFoundError("No Excel files found in the images path.")

    # Thresholds
    t_min = args.t_min
    t_max = args.t_max

    chi2_distances = []
    
    # Determine the number of plots
    num_files = len(excel_files)
    if num_files == 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(num_files // 5, 5, figsize=(20, 8))
        axes = axes.flatten()

    for i, k in enumerate(tqdm(list(excel_files.keys()), desc="Processing experiments")):
        try:
            # Read the ground truth excel
            excel_file = read_excel(excel_files[k][0])
            if excel_file.empty:
                raise ValueError(f"Excel file {excel_files[k][0]} is empty or corrupted.")

            # Calculate the ratio for converting area from px to nm
            ratio = np.mean(excel_file.iloc[:, 2]) / np.mean(excel_file.iloc[:, 1])
            if np.isnan(ratio) or ratio <= 0:
                raise ValueError(f"Invalid ratio calculated for experiment {k}.")

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

                diams_nm = [ratio * diam_px for diam_px in diams_px]
                diams_nm = [diam_nm for diam_nm in diams_nm if t_min <= diam_nm <= t_max]
                diams_nm_predicted.extend(diams_nm)

            # Generate histograms and calculate chi-squared distance
            histogram_predicted = generate_diam_nm_histogram(diams_nm_predicted, bins=[0.01, 10, 20.01, 30, 40.01, 50, 60.01, 70, 80.01, 90, 100.01, 110, 120.01, 130, 140.01, 150])
            weights_pred = histogram_predicted[0] / np.sum(histogram_predicted[0])

            histogram_gt = generate_diam_nm_histogram(excel_file.iloc[:, 2].dropna(), bins=[0.01, 10, 20.01, 30, 40.01, 50, 60.01, 70, 80.01, 90, 100.01, 110, 120.01, 130, 140.01, 150])
            weights_gt = histogram_gt[0] / np.sum(histogram_gt[0])

            chi2_dist = chi2_distance(weights_gt, weights_pred)
            chi2_distances.append(chi2_dist)

            # Save the predicted diameters to csv file
            predicted_diameters_df = pd.DataFrame(diams_nm_predicted, columns=["Diameter (nm)"])
            predicted_diameters_df.to_csv(os.path.join(args.save_path, f"{k}_predicted_diameters.csv"), index=False)

            # Plot histogram
            ax = axes[i]
            ax.hist(histogram_gt[1][:-1], bins=histogram_gt[1], weights=weights_gt, alpha=0.5, align='mid', color='blue')
            ax.hist(histogram_predicted[1][:-1], bins=histogram_predicted[1], weights=weights_pred, alpha=0.5, align='mid', color='red')
            ax.set_title(f'{k} (Chi2 distance: {chi2_dist:.4f})')
            ax.legend([f'{k} - GT', f'{k} - MASKS'], loc='upper right')
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

    mean_chi2_dist = np.mean(chi2_distances)
    print(f'Mean Chi2 Distance: {mean_chi2_dist:.4f}')

if __name__ == "__main__":
    args = parse_args()
    main(args)
