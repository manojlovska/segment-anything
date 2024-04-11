import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from PIL import Image
import collections
import pandas as pd
from skimage.measure import regionprops
from scipy import ndimage
from tqdm import tqdm
import math


def find_excel(file_path):
    xlsx_paths = sorted(glob.glob(os.path.join(file_path, '**/*.xlsx'),
                         recursive = True))

    xlsx_files = collections.defaultdict(list)

    for xlsx_path in xlsx_paths:
        dir = os.path.dirname(xlsx_path).split('/')[-1]
        xlsx_files[dir].append(xlsx_path)

    return xlsx_files

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

def read_area(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    ids = df['id'].tolist()
    areas = df['area'].tolist()

    res = dict(zip(ids, areas))

    return res

def get_diam_from_area(csv_file):
    csv_dir = os.path.join(os.path.dirname(csv_file).split('/')[-2], os.path.dirname(csv_file).split('/')[-1])

    # Read the CSV file
    df = pd.read_csv(csv_file)

    ids = df['id'].tolist()
    areas = df['area'].tolist()

    diams = [2 * math.sqrt(area / math.pi) for area in areas]

    if csv_dir.startswith('BSHF') or csv_dir.startswith('TEM'):
        diams = [diam*3 for diam in diams]

    res = dict(zip(ids, diams))

    return res


def load_masks(path, restrict_area=False, t_min = 1000.0, t_max = 30000.0):
    masks_paths = sorted(glob.glob(os.path.join(path, '**/*.png'),
                         recursive = True))

    masks = collections.defaultdict(list)

    if restrict_area:
        metadata_files = sorted(glob.glob(os.path.join(path, '**/*.csv'),
                         recursive = True))
        
        for metadata_file in metadata_files:
            dir = os.path.join(os.path.dirname(metadata_file).split('/')[-2], os.path.dirname(metadata_file).split('/')[-1])
            id_area_dict = read_area(metadata_file)

            ids = sorted([id for id, area in id_area_dict.items() if area > t_min and area < t_max])

            for id in ids:
                mask_path = os.path.join(os.path.dirname(metadata_file), f'{id}.png')
                # mask = Image.open(mask_path)
                masks[dir].append((mask_path, os.path.basename(mask_path)))
                
    else:
        for masks_path in masks_paths:
            dir = os.path.join(os.path.dirname(masks_path).split('/')[-2], os.path.dirname(masks_path).split('/')[-1])

            # mask = Image.open(masks_path)

            masks[dir].append((masks_path, os.path.basename(masks_path)))

    return masks

def load_masks_diam_restrict(path, excel_path, restrict_diam=False, t_min = 10, t_max = 150):

    masks = collections.defaultdict(list)

    if restrict_diam:
        excel_paths = find_excel(excel_path)

        for i, k in enumerate(excel_paths.keys()):
            dir = excel_paths[k][0].split('/')[-2]

            excel = read_excel(excel_paths[k][0])

            ratio = np.mean(excel.iloc[:, 2]) / np.mean(excel.iloc[:,1])

            metadata_files = sorted(glob.glob(os.path.join(path, dir, '**/*.csv'),
                                              recursive = True))
            
            for metadata in metadata_files:

                metadata_dir = os.path.join(os.path.dirname(metadata).split('/')[-2], os.path.dirname(metadata).split('/')[-1])
                id_diam_px_dict = get_diam_from_area(metadata)

                ids = sorted([id for id, diam in id_diam_px_dict.items() if (diam*ratio) > t_min and (diam*ratio) < t_max])

                for id in ids:
                    mask_path = os.path.join(os.path.dirname(metadata), f'{id}.png')
                    masks[metadata_dir].append((mask_path, os.path.basename(mask_path)))

    else:
        masks_paths = sorted(glob.glob(os.path.join(path, '**/*.png'),
                         recursive = True))
        
        for masks_path in masks_paths:
            dir = os.path.join(os.path.dirname(masks_path).split('/')[-2], os.path.dirname(masks_path).split('/')[-1])
            masks[dir].append((masks_path, os.path.basename(masks_path)))

    return masks

# Split the masks
def split_mask_v1(mask):
    thresh = mask.copy().astype(np.uint8)
    contours, _ = cv2.findContours(thresh, 2, 1)
    i = 0 
    for contour in contours:
        if  cv2.contourArea(contour) > 20:
            hull = cv2.convexHull(contour, returnPoints = False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                continue
            points = []
            dd = []

            #
            # In this loop we gather all defect points 
            # so that they can be filtered later on.
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                d = d / 256
                dd.append(d)

            for i in range(len(dd)):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                if dd[i] > 1.0 and dd[i]/np.max(dd) > 0.2:
                    points.append(f)

            i = i + 1
            if len(points) >= 2:
                for i in range(len(points)):
                    f1 = points[i]
                    p1 = tuple(contour[f1][0])
                    nearest = None
                    min_dist = np.inf
                    for j in range(len(points)):
                        if i != j:
                            f2 = points[j]                   
                            p2 = tuple(contour[f2][0])
                            dist = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) 
                            if dist < min_dist:
                                min_dist = dist
                                nearest = p2

                    cv2.line(thresh,p1, nearest, [0, 0, 0], 2)
    return thresh

def create_separate_masks(mask):

    label_im, nb_labels = ndimage.label(mask) 

    separated_masks = []
    for i in range(nb_labels):

        # create an array which size is same as the mask but filled with 
        # values that we get from the label_im. 
        # If there are three masks, then the pixels are labeled 
        # as 1, 2 and 3.

        mask_compare = np.full(np.shape(label_im), i+1) 

        # check equality test and have the value 1 on the location of each mask
        separate_mask = np.equal(label_im, mask_compare).astype(int) 

        # replace 1 with 255 for visualization as rgb image

        separate_mask[separate_mask == 1] = 255

        separated_masks.append(separate_mask)
    
    return separated_masks

def main(masks_path, output_path, excel_path, restrict_diam: bool, save_masks: bool, show_masks: bool, conv_thresh=1.1):
    all_masks = load_masks_diam_restrict(masks_path, excel_path, restrict_diam=restrict_diam)

    corrected_masks = collections.defaultdict(list)
    splitted_masks = collections.defaultdict(list)

    for key in tqdm(list(all_masks.keys())):
        img_masks = all_masks[key]

        num_rows = len(img_masks) // 8 + (1 if len(img_masks) % 8 != 0 else 0)
        # Create a figure
        fig, axes = plt.subplots(num_rows, 8)

        if num_rows == 1:
            axes = axes.reshape(1, -1)

        fig.set_figwidth(40)
        fig.set_figheight(5*num_rows)

        for idx, (mask_path, mask_name) in enumerate(img_masks):
            img_mask = Image.open(mask_path)
            i = idx // 8
            j = idx % 8
            mask = np.array(img_mask)

            props = regionprops(mask, cache=False)
            prop = props[0]

            if prop.convex_area/prop.filled_area > conv_thresh:
                splitted_mask = split_mask_v1(np.array(mask))
                separated_masks = create_separate_masks(splitted_mask)

                if save_masks:
                    for m, sep_mask in enumerate(separated_masks):
                        dir = os.path.join(output_path, key)
                        os.makedirs(dir, exist_ok=True)
                        save_filename = os.path.join(dir, mask_name.split('.')[-2] + f'_splitted_{m}.png')
                        
                        save_mask = Image.fromarray(sep_mask.astype(np.uint8))
                        save_mask.save(save_filename)

                # Save the splitted masks separately, and all masks together
                splitted_masks[key].append(splitted_mask)
                corrected_masks[key].append(splitted_mask)

                # Plot the splitted mask
                
                plotting_mask = splitted_mask
                axes[i][j].set_title(f'$\mathbf{{{mask_name}: non-convex}}$', fontsize=25)
            
            else:
                if save_masks:
                    dir = os.path.join(output_path, key)
                    os.makedirs(dir, exist_ok=True)

                    save_filename = os.path.join(dir, mask_name)

                    save_mask = Image.fromarray(mask.astype(np.uint8))
                    save_mask.save(save_filename)

                corrected_masks[key].append(mask)

                plotting_mask = mask

                axes[i][j].set_title(f'{mask_name}: convex', fontsize=25)

            
            axes[i][j].imshow(plotting_mask)
            img_mask.close()

        
        if len(img_masks) % 8 != 0:
            for ax in axes.flatten()[len(img_masks):]:
                fig.delaxes(ax)

        plt.suptitle(f'{key}', fontsize=30, fontweight='bold')
        plt.tight_layout(w_pad=2.5)

        if save_masks:
            plt.savefig(os.path.join(dir, os.path.basename(dir) + '_convexity.png'))
        if show_masks:
            plt.show()

        plt.close()



if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)

    masks_path = '/work/anastasija/Materials-Science/segment-anything/output'
    output_path = '/work/anastasija/Materials-Science/segment-anything/output-splitted-t_0.95-prefiltered'
    excel_path = '/work/anastasija/Materials-Science/Datasets/Porazdelitev celcev original/'

    main(masks_path, output_path, excel_path, restrict_diam=True, save_masks=True, show_masks=False, conv_thresh=1.1)

