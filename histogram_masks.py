import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import fnmatch
import math

def generate_diam_nm_histogram(data):
    # Define custom bins
    bins = [0.01, 10,
            20.01, 30,
            40.01, 50,
            60.01, 70,
            80.01, 90,
            100.01, 110,
            120.01, 130,
            140.01, 240]

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

def read_area(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    ids = df['id']
    areas = df['area']

    return ids, areas

def calculate_diam(area):
    diam = 2 * math.sqrt(area / math.pi)

    return diam

output_file_path = '/work/anastasija/Materials-Science/segment-anything/output'

particle_file_path = '/work/anastasija/Materials-Science/segment-anything/output/TEM BSHF-DBSA-210325'

csv_files = sorted(find_csv(particle_file_path))

area_diam_diam_nm = {}
diam_nm = {}
all_diams_nm = []
ratio = 0.119727

for idx, csv in enumerate(csv_files):
    ids, areas = read_area(csv)
    ids = ids.tolist()
    areas = areas.tolist()
    diams = [calculate_diam(area) for area in areas]
    diams_nm = [diam*ratio for diam in diams]
    area_diam_diam_nm[idx] = dict(zip(ids, [[areas[i], diams[i], diams_nm[i]] for i in range(len(areas))]))
    diam_nm[idx] = dict(zip(ids, diams_nm))

    all_diams_nm.extend(diams_nm)

histogram = generate_diam_nm_histogram(all_diams_nm)

plt.hist(histogram[1].tolist()[:-1], histogram[1].tolist()[:-1], weights=histogram[0].tolist(), align='mid', density=True)
plt.title('Histogram')
plt.xlabel('Premer (nm)')
plt.ylabel('Št. delcev')
plt.xticks(histogram[1].tolist()[:-1])
plt.grid(True)
plt.show()






max_area = 441223.0




