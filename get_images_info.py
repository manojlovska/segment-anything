import os
import json
import pandas as pd
import numpy as np
import math
import glob
import collections
from PIL import Image

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

def find_excel(file_path):
    xlsx_paths = sorted(glob.glob(os.path.join(file_path, '**/*.xlsx'),
                         recursive = True))

    xlsx_files = collections.defaultdict(list)

    for xlsx_path in xlsx_paths:
        dir = xlsx_path.split('/')[-2]
        xlsx_files[dir].append(xlsx_path)

    return xlsx_files

def get_ratio_from_excel(excel_path):
    excel_file = read_excel(excel_path)

    ratio = np.mean(excel_file.iloc[:, 2]) / np.mean(excel_file.iloc[:,1])

    return ratio

def get_all_ratios(root_dir):
    xlsx_paths = find_excel(root_dir)

    ratios_dict = {}
    for key, value in xlsx_paths.items():
        ratio = get_ratio_from_excel(value[0])

        ratios_dict[key] = ratio

    return ratios_dict


root_dir = '/home/matejm/anastasija/Materials-Science/segment-anything/data/Porazdelitev celcev original'
images_paths = sorted([file for file in glob.glob(os.path.join(root_dir, '**', '*.tif'), recursive=True) 
                       if not (file.endswith('-a.tif') or file.endswith('-1.tif'))])
save_path = '/home/matejm/anastasija/Materials-Science/segment-anything/data/'

ratios_dict = get_all_ratios(root_dir)

info_dict = {}
for id, image_path in enumerate(images_paths):
    dir = image_path.split('/')[-2]
    dict_key = os.path.join(image_path.split('/')[-2], image_path.split('/')[-1])

    image_size = Image.open(image_path).size[::-1]
    ratio = ratios_dict[dir]

    if id < 148:
        split = "train"
    else:
        split = "val"

    info_dict[dict_key] = {"id": id, 
                           "image_size": image_size, 
                           "r": ratio,
                           "split": split}

# Save the dictionary in json file
with open(os.path.join(save_path, "dataset_info.json"), "w") as json_file:
    json.dump(info_dict, json_file)


