import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import fnmatch

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

file_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/TEM BSHF-DBSA-210325'

excel_files = []
for dirpath, dirnames, filenames in os.walk(file_path):
    for filename in fnmatch.filter(filenames, '*.xlsx'):
        excel_files.append(os.path.join(dirpath, filename))


for excel_file in excel_files:
    excel = read_excel(excel_file)

    max_area = find_max_area(excel)

    histogram = generate_diam_nm_histogram(excel.iloc[:, 2].dropna())
    weights = histogram[0].tolist()
    normalized_weights = np.array(weights) / np.sum(weights)

    # Plot histogram
    plt.hist(histogram[1].tolist()[:-1], bins=histogram[1].tolist(), weights=normalized_weights, align='mid')
    plt.xlabel('Premer (nm)')
    plt.ylabel('Št. delcev')
    plt.xticks(histogram[1].tolist()[:-1])
    plt.grid(True)
    plt.show()