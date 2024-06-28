import numpy as np
import pandas as pd
import os
import math
import json
import collections
import glob
from pycocotools import mask
import cv2
import fnmatch

def im_clear_borders(thresh):
    """
    Remove elements attached to the image border.

    Parameters:
    thresh (numpy.ndarray): The binary image (thresholded).

    Returns:
    numpy.ndarray: The binary image with border elements removed.
    """
    kernel2 = np.ones((3,3), np.uint8)
    marker = thresh.copy()
    marker[1:-1,1:-1] = 0
    while True:
        tmp = marker.copy()
        marker = cv2.dilate(marker, kernel2)
        marker = cv2.min(thresh, marker)
        difference = cv2.absdiff(marker, tmp)
        if cv2.countNonZero(difference) == 0:
            break
    mask = cv2.bitwise_not(marker)
    out = cv2.bitwise_and(thresh, mask)
    return out

def refine_masks(mask):
    """
    Refine masks by removing elements attached to the image border.

    Parameters:
    mask (numpy.ndarray): The binary mask.

    Returns:
    numpy.ndarray: The refined binary mask.
    """
    # https://stackoverflow.com/questions/65534370/remove-the-element-attached-to-the-image-border
    mask = im_clear_borders(mask.astype(np.uint8)).astype(bool)
    return mask

def decode_annotation(json_file):
    """
    Decodes annotations from a JSON file (output of the SAM model).

    Parameters:
    json_file (str): Path to the JSON file containing annotations.

    Returns:
    dict: A dictionary with the following keys:
        - binary_masks: List of decoded binary masks.
        - areas: List of areas for each annotation.
        - bboxes: List of bounding boxes for each annotation.
        - predicted_ious: List of predicted IOUs for each annotation.
        - point_coords: List of point coordinates for each annotation.
        - stability_scores: List of stability scores for each annotation.
        - crop_boxes: List of crop boxes for each annotation.
    """

    with open(json_file, 'r') as f:
        annotations = json.load(f)

    coco_rles = [annotations[i]['segmentation'] for i in range(len(annotations))]
    binary_masks = [mask.decode(coco_rles[i]) for i in range(len(coco_rles))]
    areas = [annotations[i]['area'] for i in range(len(annotations))]
    bboxes = [annotations[i]['bbox'] for i in range(len(annotations))]
    predicted_ious = [annotations[i]['predicted_iou'] for i in range(len(annotations))]
    point_coords = [annotations[i]['point_coords'] for i in range(len(annotations))]
    stability_scores = [annotations[i]['stability_score'] for i in range(len(annotations))]
    crop_boxes = [annotations[i]['crop_box'] for i in range(len(annotations))]

    result = {"binary_masks": binary_masks, 
              "areas": areas, 
              "bboxes": bboxes, 
              "predicted_ious": predicted_ious, 
              "point_coords": point_coords, 
              "stability_scores": stability_scores, 
              "crop_boxes": crop_boxes}

    return result

def generate_diam_nm_histogram(data, bins):
    """
    Generates a histogram for diameter data in nanometers using custom bins.

    Parameters:
    data (list): List of diameter values in nanometers.

    Returns:
    tuple: A tuple containing the histogram counts and bin edges.
    """
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

def find_files(file_path, file_extension):
    """
    Finds all files with a given extension in a directory and its subdirectories.

    Parameters:
    file_path (str): The root directory to search for files.
    file_extension (str): The file extension to search for (e.g., '.json', '.xlsx').

    Returns:
    dict: A dictionary where keys are directory names and values are lists of file paths.
    """
    file_paths = sorted(glob.glob(os.path.join(file_path, f'**/*{file_extension}'), recursive=True))
    files_dict = collections.defaultdict(list)

    for path in file_paths:
        dir_name = os.path.basename(os.path.dirname(path))
        files_dict[dir_name].append(path)

    return files_dict

def find_images(images_path, file_extension):
    """
    Finds images with the specified file extension in a given directory and its subdirectories,
    excluding files that end with '-a.<extension>' or '-1.<extension>'.

    Parameters:
    images_path (str): The root directory to search for image files.
    file_extension (str): The file extension to search for (e.g., 'tif' or 'jpg').

    Returns:
    list: A list of file paths for the images found.
    """
    # List to store the paths of the images
    image_files = []

    # Ensure the file extension starts with a dot
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(images_path):
        # Filter out files that match the specified extension but do not end with '-a.<extension>' or '-1.<extension>'
        for filename in fnmatch.filter(filenames, '*' + file_extension):
            if not (filename.endswith('-a' + file_extension) or filename.endswith('-1' + file_extension)):
                image_files.append(os.path.join(dirpath, filename))

    return image_files

def calculate_diam(area):
    """
    Calculates the diameter of a circle given its area.

    Parameters:
    area (float): The area of the circle.

    Returns:
    float: The diameter of the circle.
    """
    diam = 2 * math.sqrt(area / math.pi)

    return diam

def read_excel(file_path):
    """
    Reads an Excel file and processes its content.

    Parameters:
    file_path (str): Path to the Excel file.

    Returns:
    DataFrame: A pandas DataFrame with processed content. If the first column contains numbers,
               a header is added. Otherwise, the first row is used as the header.
    """
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

def chi2_distance(A, B):
    """
    Computes the chi-squared distance between two distributions.

    Parameters:
    A (list): The first distribution.
    B (list): The second distribution.

    Returns:
    float: The chi-squared distance between the two distributions.
    """
    chi = 0.0
    for a, b in zip(A, B):
        if a + b != 0:
            chi += ((a - b) ** 2) / (a + b)
    return 0.5 * chi

def check_image_extension(file_path):
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    
    # Check if it ends with '.tif' or '.jpg'
    if file_extension.lower() in ['.tif', '.jpg']:
        return True
    else:
        return False
