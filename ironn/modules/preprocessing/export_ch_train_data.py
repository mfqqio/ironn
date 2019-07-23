import os
import matplotlib.pyplot as plt
import pandas as pd
import ast
import shutil
import argparse
from glob import glob
import numpy as np
import cv2
from get_roughness import ImageColorMiner

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, required=True,
                    help='input file path for the convex hull coordinates')
parser.add_argument('--data_dir', type=str, required=True,
                    help='input directory path for original images')
parser.add_argument('--train_dir', type=str, required=True,
                    help='directory path for training images')
parser.add_argument('--label_dir', type=str, required=True,
                    help='directory path for exporting labels')
parser.add_argument('--test_dir', type=str, required=True,
                    help='directory path for test images')
parser.add_argument('--superimpose_dir', type=str,
                    help='directory path for exporting images superimposed with labels')
parser.add_argument('--num_test', type=int, default=50,
                    help='number of new unseen images for the test set')

args = parser.parse_args()

def read_data(input_file, data_dir):
    """
    Read the convex hull coordinates

    Parameters:
    -----------
    input_file: str
            path to the input csv file with convex hull coordinates
    data_dir: str
            path to the directory with original images
    Returns:
    --------
    ch: pd.DataFrame
        a pandas DataFrame containing filenames and hull coordinates
    image_list: list
        a list of jpeg image paths from the data_dir
    """
    ch = pd.read_csv(input_file, index_col=0)
    ch.columns = ['filename', 'hull_coordinates']

    image_list = glob(os.path.join(data_dir, "*.jpg"))
    image_list.extend(glob(os.path.join(data_dir, '*.JPG')))

    return ch, image_list

def get_label_dict(ch, image_list, data_dir):
    """
    Gets a dictionary of images with their labels (masks)

    Parameters:
    -----------
    ch: pd.DataFrame
            a pandas DataFrame containing filenames and hull coordinates
    image_list: list
            a list of jpeg image paths from the data_dir
    data_dir: str
            path to the directory with original images
    Returns:
    --------
    labels: dict
        a dictionary of image names and their label masks
    """
    labels = {}
    icl = ImageColorMiner(img_folder_path = data_dir)
    for im in image_list:
        if "validate" not in im:
            image_path = os.path.join(data_dir, im)
            image = cv2.imread(image_path)
            jsonfile = im.split("/")[-1].split(".")[0] + ".json"
            hull = ch.loc[ch.filename == jsonfile]['hull_coordinates']
            if not hull.empty:
                hull = hull.tolist()
                hull = ast.literal_eval(hull[0])
                try:
                    mask, out = icl.extract_aoi(im.split("/")[-1], hull)
                    labels[im] = mask
                except ValueError:
                    print(f"Error in getting the mask for {im}, ignoring it.")

    return labels

def export_labels(labels, label_dir):
    """
    Exports the label masks to image files

    Parameters:
    -----------
    labels: dict
            a dictionary of image names and their label masks
    label_dir: str
            path to the directory for saving label images
    Returns:
    --------
    None
    """
    for key, value in labels.items():
        if not os.path.exists(label_dir):
            os.mkdir(label_path)
        filepath = os.path.join(label_dir, key.split("/")[-1])
        mask = value
        mask = mask.reshape(*mask.shape, 1)
        mask = np.concatenate((mask, np.invert(mask)), axis=2)
        mask = np.concatenate((mask, np.zeros((*mask[:,:,0].shape, 1))), axis=2)
        plt.imsave(filepath, mask)

def make_train_test(labels, data_dir, train_dir, test_dir, num_test):
    """
    Getting images with labels into a separate directory

    Parameters:
    -----------
    labels: dict
            a dictionary of image names and their label masks
    data_dir: str
            path to the directory with original images
    train_dir: str
            path to the directory for saving training images
    test_dir: str
            path to the directory for saving test images
    num_test: int
            number of test images to save out of the total images
    Returns:
    --------
    None
    """
    for key, value in labels.items():
        filename = os.path.join(data_dir, key)
        if os.path.isfile(filename):
            shutil.copy(filename, train_dir)

    all_images = glob(os.path.join(data_dir, '*.jpg'))
    all_images.extend(glob(os.path.join(data_dir, "*.JPG")))

    i = 0
    for img in all_images:
        i+=1
        if i == num_test:
            break
        if img not in labels.keys():
            shutil.copy(img, test_dir)

def superimpose(train_dir, label_dir, superimpose_dir):
    """
    Superimposing and saving iamges with label masks

    Parameters:
    -----------
    train_dir: str
            path to the directory for saving training images
    label_dir: str
            path to the directory for saving label images
    superimpose_dir: str
            path to the directory for saving superimposed images
    Returns:
    --------
    None
    """
    image_list = glob(os.path.join(train_dir, "*.jpg"))
    image_list.extend(glob(os.path.join(train_dir, '*.JPG')))
    label_list = glob(os.path.join(label_dir, "*.jpg"))
    label_list.extend(glob(os.path.join(label_dir, '*.JPG')))
    for im in image_list:
        image_name = im.split("/")[-1]
        image_path = os.path.join(train_dir, image_name)
        label_path = os.path.join(label_dir, image_name)
        if os.path.isfile(image_path) and os.path.isfile(label_path):
            image = cv2.imread(image_path)
            label = cv2.imread(label_path)
            if image.shape == label.shape:
                output = cv2.addWeighted(image, 0.7, label, 0.3, 0, dtype = 0)
                plt.imsave(os.path.join(superimpose_dir, image_name), output)
            else:
                print(f"This image doesn't have a compatible sized label {image_name}")
        else:
            print(f"One of the image and the label doesn't exist: {image_path}, {label_path}")

def main():
    if not os.path.exists(args.label_dir):
        os.makedirs(args.label_dir)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    print("Reading in the convex hull data..")
    ch, image_list = read_data(args.input_file, args.data_dir)
    print("Extracting label information..")
    labels = get_label_dict(ch, image_list, args.data_dir)
    print("Exporting label images..")
    export_labels(labels, args.label_dir)
    print("Making directories with training images and labels..")
    make_train_test(labels, args.data_dir, args.train_dir, args.test_dir, args.num_test)
    if args.superimpose_dir:
        print("Superimposing the images and labels..")
        if not os.path.exists(args.superimpose_dir):
            os.makedirs(args.superimpose_dir)
        superimpose(args.train_dir, args.label_dir, args.superimpose_dir)
    print("Process completed!")

if __name__ == "__main__":
    main()
