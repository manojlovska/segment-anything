import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import fnmatch
import math
import json
from tqdm import tqdm


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
    csv_files = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(file_path):
        # Filter out PNG files that do not end with '-a.png'
        for filename in fnmatch.filter(filenames, '*.csv'):
            csv_files.append(os.path.join(dirpath, filename))

    return csv_files

def find_json(file_path):
    json_files = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(file_path):
        # Filter out PNG files that do not end with '-a.png'
        for filename in fnmatch.filter(filenames, '*.json'):
            json_files.append(os.path.join(dirpath, filename))

    return json_files

def read_area(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    ids = df['id']
    areas = df['area']

    return ids, areas

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

def read_json(file_name):
    # Opening JSON file
    f = open(file_name)
 
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    return data

def read_area_json(json_file):
    json_dict = read_json(json_file)

    ids = []
    areas = []
    for i in range(len(json_dict)):
        ids.append(i)
        areas.append(json_dict[i]['area'])

    return ids, areas

############################################################################ MAIN
#############################################################################################################################################
################################################### JSON - FOR ALL THE AREAS, NO FILTERING

data_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/' # TEM BSHF-DBSA-210325'
masks_path = '/work/anastasija/Materials-Science/segment-anything/output'

excel_files = []
for dirpath, dirnames, filenames in os.walk(data_path):
    for filename in fnmatch.filter(filenames, '*.xlsx'):
        excel_files.append(os.path.join(dirpath, filename))

chi2_distances = []
_,axes = plt.subplots(3,5, figsize=(40,20))
for i, excel_file in enumerate(sorted(excel_files)):
    print(f'Excel file [{i+1}/{len(excel_files)}]: {os.path.basename(excel_file)}, \nFolder [{i+1}/{len(excel_files)}]: {excel_file.split("/")[-2]}')
    row = i // 5
    col = i % 5
    excel = read_excel(excel_file)
    max_area = find_max_area(excel)

    area_diam_diam_nm = {}
    diam_nm = {}
    all_diams_nm = []
    ratio = np.mean(excel.iloc[:, 2]) / np.mean(excel.iloc[:,1])

    json_files = sorted(find_json(os.path.join(masks_path, excel_file.split('/')[-2])))
    for idx, json_file in tqdm(zip(range(len(json_files)), json_files), total=len(json_files)):
        ids, areas = read_area_json(json_file)

        diams = [calculate_diam(area) for area in areas]
        diams_nm = [diam*ratio for diam in diams]
        area_diam_diam_nm[idx] = dict(zip(ids, [[areas[i], diams[i], diams_nm[i]] for i in range(len(areas))]))
        diam_nm[idx] = dict(zip(ids, diams_nm))

        

        all_diams_nm.extend(diams_nm)

    histogram_masks = generate_diam_nm_histogram(all_diams_nm)
    weights_masks = histogram_masks[0].tolist()
    normalized_weights_masks = np.array(weights_masks) / np.sum(weights_masks)

    if excel_file.split('/')[-2].startswith('BSHF') or excel_file.split('/')[-2].startswith('TEM'):
        gt_diam_px_excel = excel.iloc[:, 1].dropna()
        gt_diam_px_scaled = [gt_diam / 3 for gt_diam in gt_diam_px_excel]                               ##### Scaling factor is 3
        gt_diam_nm = [gt_diam_px * ratio for gt_diam_px in gt_diam_px_scaled]
    
    else:
        gt_diam_nm = excel.iloc[:, 2].dropna()


    histogram_gt = generate_diam_nm_histogram(gt_diam_nm)
    weights_gt = histogram_gt[0].tolist()
    normalized_weights_gt = np.array(weights_gt) / np.sum(weights_gt)

    chi2_dist = chi2_distance(normalized_weights_gt, normalized_weights_masks)
    chi2_distances.append(chi2_dist)

    # Plot histogram
    axes[row][col].hist(histogram_gt[1].tolist()[:-1], bins=histogram_gt[1].tolist(), weights=normalized_weights_gt, alpha=0.5, align='mid', color='blue')
    axes[row][col].hist(histogram_masks[1].tolist()[:-1], bins=histogram_masks[1].tolist(), weights=normalized_weights_masks, alpha=0.5, align='mid', color='red')
    axes[row][col].set_title(excel_file.split('/')[-2] + f' (Chi2 distance: {chi2_dist})')
    axes[row][col].legend([excel_file.split('/')[-2] + ' - GT', excel_file.split('/')[-2] + ' - MASKS'], loc='upper right')
    axes[row][col].set_xlabel('Premer (nm)')
    axes[row][col].set_ylabel('Relativno št. delcev')
    axes[row][col].grid(True)
    # plt.show()

mean_chi2_dist = np.mean(chi2_distances)
plt.tight_layout()
plt.savefig(f'/work/anastasija/Materials-Science/segment-anything/output_presentation/histograms_all_areas_mean_chi2_{mean_chi2_dist}_scaled.png')
plt.show()

import pdb
pdb.set_trace()

#############################################################################################################################################
################################################### CSV FILTERED
data_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/' # TEM BSHF-DBSA-210325'
masks_path = '/work/anastasija/Materials-Science/segment-anything/output'

t_1 = 1000.0
t_2 = 250000.0

excel_files = []
for dirpath, dirnames, filenames in os.walk(data_path):
    for filename in fnmatch.filter(filenames, '*.xlsx'):
        excel_files.append(os.path.join(dirpath, filename))

chi2_distances = []
_,axes = plt.subplots(3,5, figsize=(40,20))
for i, excel_file in enumerate(sorted(excel_files)):
    row = i // 5
    col = i % 5
    excel = read_excel(excel_file)
    max_area = find_max_area(excel)

    area_diam_diam_nm = {}
    diam_nm = {}
    all_diams_nm = []
    ratio = np.mean(excel.iloc[:, 2]) / np.mean(excel.iloc[:,1])
    csv_files = find_csv(os.path.join(masks_path, excel_file.split('/')[-2]))
    for idx, csv in enumerate(csv_files):
        ids, areas = read_area(csv)
        ids = ids.tolist()
        areas = areas.tolist()
        indices = [index for index, area in enumerate(areas) if area < t_1 or area > t_2]

        # Do not take into account areas bigger than t_1 and smaller than t_2
        if indices:
            for index in sorted(indices, reverse=True):
                del areas[index]
                del ids[index]



        diams = [calculate_diam(area) for area in areas]
        diams_nm = [diam*ratio for diam in diams]
        area_diam_diam_nm[idx] = dict(zip(ids, [[areas[i], diams[i], diams_nm[i]] for i in range(len(areas))]))
        diam_nm[idx] = dict(zip(ids, diams_nm))

        

        all_diams_nm.extend(diams_nm)

    histogram_masks = generate_diam_nm_histogram(all_diams_nm)
    weights_masks = histogram_masks[0].tolist()
    normalized_weights_masks = np.array(weights_masks) / np.sum(weights_masks)

    if excel_file.split('/')[-2].startswith('BSHF') or excel_file.split('/')[-2].startswith('TEM'):
        gt_diam_px_excel = excel.iloc[:, 1].dropna()
        gt_diam_px_scaled = [gt_diam / 3 for gt_diam in gt_diam_px_excel]##### Scaling factor is 3
        gt_diam_nm = [gt_diam_px * ratio for gt_diam_px in gt_diam_px_scaled]
    
    else:
        gt_diam_nm = excel.iloc[:, 2].dropna()

    histogram_gt = generate_diam_nm_histogram(gt_diam_nm)
    weights_gt = histogram_gt[0].tolist()
    normalized_weights_gt = np.array(weights_gt) / np.sum(weights_gt)

    chi2_dist = chi2_distance(normalized_weights_gt, normalized_weights_masks)
    chi2_distances.append(chi2_dist)

    # Plot histogram
    axes[row][col].hist(histogram_gt[1].tolist()[:-1], bins=histogram_gt[1].tolist(), weights=normalized_weights_gt, alpha=0.5, align='mid', color='blue')
    axes[row][col].hist(histogram_masks[1].tolist()[:-1], bins=histogram_masks[1].tolist(), weights=normalized_weights_masks, alpha=0.5, align='mid', color='red')
    axes[row][col].set_title(excel_file.split('/')[-2] + f' (Chi2 distance: {chi2_dist})')
    axes[row][col].legend([excel_file.split('/')[-2] + ' - GT', excel_file.split('/')[-2] + ' - MASKS'], loc='upper right')
    axes[row][col].set_xlabel('Premer (nm)')
    axes[row][col].set_ylabel('Relativno št. delcev')
    axes[row][col].grid(True)
    # plt.show()

plt.tight_layout()
plt.savefig('/work/anastasija/Materials-Science/segment-anything/output_presentation/histograms2.png')
plt.show()

mean_chi2_dist = np.mean(chi2_distances)
print(mean_chi2_dist)

