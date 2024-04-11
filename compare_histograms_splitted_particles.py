import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import fnmatch
import math
import json
from tqdm import tqdm
import collections
import glob
from PIL import Image


def generate_diam_nm_histogram(data):
    # Define custom bins
    bins = [0.01, 10,
            20.01, 30,
            40.01, 50,
            60.01, 70,
            80.01, 90,
            100.01, 110,
            120.01, 130,
            140.01, 150]

    hist = np.histogram(data, bins=bins)

    return hist

def find_csv(file_path):
    csv_paths = sorted(glob.glob(os.path.join(file_path, '**/*.csv'),
                         recursive = True))

    csv_files = collections.defaultdict(list)

    for csv_path in csv_paths:
        dir = os.path.join(os.path.dirname(csv_path).split('/')[-2], os.path.dirname(csv_path).split('/')[-1])
        csv_files[dir].append(csv_path)

    return csv_files

def find_excel(file_path):
    xlsx_paths = sorted(glob.glob(os.path.join(file_path, '**/*.xlsx'),
                         recursive = True))

    xlsx_files = collections.defaultdict(list)

    for xlsx_path in xlsx_paths:
        dir = os.path.dirname(xlsx_path).split('/')[-1]
        xlsx_files[dir].append(xlsx_path)

    return xlsx_files

def read_areas(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    ids = df['id']
    areas = df['area']

    return ids, areas

def get_area_from_id(id, df):
    area = df.loc[df["id"] == id, "area"]

    return int(area.iloc[0])


def calculate_diam(area):
    diam = 2 * math.sqrt(area / math.pi)

    return diam

def read_excel(file_path):
    excel = pd.read_excel(file_path, usecols=[0,1,2], header=None)

    # Check if the first column contains numbers
    if  any(excel.dtypes == 'float64'):
        # If the first column contains numbers, add a header
        excel.columns = ['area (pixel)', 'premer (pixel)', 'premer (nm)']
    else:
        # Extract the first column as the header
        header = excel.iloc[0]

        # Remove the first column from the DataFrame
        excel = excel.drop(0)

        # Assign the extracted column as the header
        excel.columns = header

    return excel

def find_max_area(excel):
    max_area = excel.iloc[:, 0].max()

    return max_area

def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.0
    for a, b in zip(A, B):
        if a + b != 0:
            chi += ((a - b) ** 2) / (a + b)
    return 0.5 * chi

def load_masks(path):
    masks_paths = sorted(glob.glob(os.path.join(path, '**/*.png'),
                         recursive = True))

    masks = collections.defaultdict(list)

    for masks_path in masks_paths:
        dir = os.path.join(os.path.dirname(masks_path).split('/')[-2], os.path.dirname(masks_path).split('/')[-1])

        masks[dir].append((masks_path, os.path.basename(masks_path)))

    return masks


# Excel path
masks_path = '/work/anastasija/Materials-Science/segment-anything/output'
csv_path = '/work/anastasija/Materials-Science/segment-anything/output'
excel_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/'

# Metadata for generated masks
csv_files = find_csv(csv_path)

# Splitted masks paths
all_masks = load_masks(masks_path)

# Excel files with ground truths
excel_files = find_excel(excel_path)

# Thresholds
t_min = 10
t_max = 150

chi2_distances = []
_,axes = plt.subplots(3,5, figsize=(40,20))
for i, k in tqdm(enumerate(list(excel_files.keys()))):
    row = i // 5
    col = i % 5

    # Read the ground truth excel
    excel_file = read_excel(excel_files[k][0])

    # Calculate the ratio for converting area from px to nm
    ratio = np.mean(excel_file.iloc[:, 2]) / np.mean(excel_file.iloc[:,1])

    # Images inside the specific experiment analysis
    keys = [k_mask for k_mask in all_masks.keys() if k_mask.startswith(k)]

    print(f' K: {k} \n Images:{keys}')

    diams_nm_predicted = []
    for key in tqdm(keys):

        img_masks = all_masks[key]

        for idx, (mask_path, mask_name) in enumerate(img_masks):
            metadata = pd.read_csv(csv_files[key][0])

            if '_' in mask_name.split('.')[0]:
                area = np.count_nonzero(np.asarray(Image.open(mask_path)) == 255)
            else:
                mask_id = int(mask_name.split('.')[0])

                area = get_area_from_id(mask_id, metadata)
            
            diam_px = calculate_diam(area)
            
            if key.startswith('BSHF') or key.startswith('TEM'):
                # Images are in smaller resolution the diameters need to be multiplied by 3 (scaling factor)
               diam_px = diam_px * 3

            diam_nm = ratio * diam_px

            if diam_nm > t_min and diam_nm < t_max:
                diams_nm_predicted.append(diam_nm)
    

    # Generate predicted histogram
    histogram_predicted = generate_diam_nm_histogram(diams_nm_predicted)
    weights_pred = histogram_predicted[0].tolist()
    normalized_weights_pred = np.array(weights_pred) / np.sum(weights_pred)

    # Generate ground truth histogram
    histogram_gt = generate_diam_nm_histogram(excel_file.iloc[:, 2].dropna())
    weights_gt = histogram_gt[0].tolist()
    normalized_weights_gt = np.array(weights_gt) / np.sum(weights_gt)

    chi2_dist = chi2_distance(normalized_weights_gt, normalized_weights_pred)
    chi2_distances.append(chi2_dist)

    # Plot histogram
    axes[row][col].hist(histogram_gt[1].tolist()[:-1], bins=histogram_gt[1].tolist(), weights=normalized_weights_gt, alpha=0.5, align='mid', color='blue')
    axes[row][col].hist(histogram_predicted[1].tolist()[:-1], bins=histogram_predicted[1].tolist(), weights=normalized_weights_pred, alpha=0.5, align='mid', color='red')
    axes[row][col].set_title(k + f' (Chi2 distance: {chi2_dist})')
    axes[row][col].legend([k + ' - GT', k + ' - MASKS'], loc='upper right')
    axes[row][col].set_xlabel('Premer (nm)')
    axes[row][col].set_ylabel('Relativno št. delcev')
    axes[row][col].grid(True)

plt.tight_layout()
# plt.savefig('/work/anastasija/Materials-Science/segment-anything/output_presentation/histograms2.png')
plt.show()

mean_chi2_dist = np.mean(chi2_distances)
print(mean_chi2_dist)
    
