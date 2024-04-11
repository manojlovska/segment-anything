import pandas
import os
import numpy as np


def get_constraints():
    folder = "/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev/unlabeled"
    all_sizes = []
    all_diameters = []
    all_size_to_diam = []
    find_max = 0
    for i in range(1,15):
        folder_path = folder + "/" + str(i)
        files = os.listdir(folder_path)
        # print([x for x in files if x.endswith('.xlsx')])
        for f in files:
            if f.endswith('.xlsx'):
                print(f)
                df = pandas.read_excel(os.path.join(folder_path, f), sheet_name='Sheet1')
                sizes = df['area (pixel)'].tolist()
                diameters = df['premer (pixel)'].tolist()
                size_to_diam = (np.array(diameters)/np.array(sizes)).tolist()
                all_sizes.extend(sizes)
                all_diameters.extend(diameters)
                for d in diameters:
                    if d > find_max:
                        print(f'd: {d}, file: {f}')
                        find_max = d
                all_size_to_diam.extend(size_to_diam)

    max_size = max(all_sizes)
    min_size = min(all_sizes)
    max_diameter = max(all_diameters)
    min_diameter = min(all_diameters)
    max_size_to_diam = max(all_size_to_diam)
    min_size_to_diam = min(all_size_to_diam)
    return (max_size, min_size, max_diameter, min_diameter, max_size_to_diam, min_size_to_diam)

max_size, min_size, max_diameter, min_diameter, max_size_to_diam, min_size_to_diam = get_constraints()