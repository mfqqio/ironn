import os
import matplotlib.pyplot as plt
import pandas as pd
import ast
import shutil
import argparse
from glob import glob
import numpy as np
import cv2
import random
from get_roughness import ImageColorMiner
from utils import give_color_to_seg_img

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, required=True,
                    help='input file path for the polygon coordinates')
parser.add_argument('--data_dir', type=str, required=True,
                    help='input directory path for original images')
parser.add_argument('--train_dir', type=str, required=True,
                    help='directory path for training images')
parser.add_argument('--label_dir', type=str, required=True,
                    help='directory path for exporting labels')
parser.add_argument('--test_dir', type=str, required=True,
                    help='directory path for test images')
parser.add_argument('--val_dir', type=str, required=True,
                    help='directory path for validation images')
parser.add_argument('--val_label_dir', type=str, required=True,
                    help='directory path for validation image labels')
parser.add_argument('--superimpose_dir', type=str,
                    help='directory path for exporting images superimposed with labels')
parser.add_argument('--num_test', type=int, default=50,
                    help='number of new unseen images for the test set')
parser.add_argument('--num_valid', type=int, default=50,
                    help='number of images to use in the validation set')

args = parser.parse_args()

''' Example usage:
python3 ubc_geodetection_review/ironn/modules/preprocessing/export_poly_train_data.py \
--input_file "ubc_geodetection_review/ironn/modules/output/img_tbl_w_labels.csv" --data_dir "04_nextgen/" \
--label_dir "training_data_all_poly/labels/" --train_dir "training_data_all_poly/training" \
--test_dir "training_data_all_poly/testing" --superimpose_dir "training_data_all_poly/superimposed/" \
--val_dir "training_data_all_poly/validation" --num_test 100
'''

def read_data(input_file, data_dir):
    """
    Read the polygon label data

    Parameters:
    -----------
    input_file: str
            path to the input csv file with polygon coordinates
    data_dir: str
            path to the directory with original images
    Returns:
    --------
    poly: pd.DataFrame
        a pandas DataFrame containing filenames and polygon coordinates
    image_list: list
        a list of jpeg image paths from the data_dir
    """
    poly = pd.read_csv(input_file, index_col=0)

    image_list = glob(os.path.join(data_dir, "*.jpg"))
    image_list.extend(glob(os.path.join(data_dir, '*.JPG')))

    return poly, image_list

def export_labels(poly, image_list, data_dir, label_dir):
    """
    Exports the label masks to image files

    Parameters:
    -----------
    poly: pd.DataFrame
        a pandas DataFrame containing filenames and polygon coordinates
    image_list: list
        a list of jpeg image paths from the data_dir
    data_dir: str
            path to the directory with original images
    label_dir: str
            path to the directory for saving label images
    Returns:
    --------
    None
    """
    labels = {}
    non_labeled = []
    for im in image_list:
        # getting the image details from the csv of labels
        image = cv2.imread(im)
        im_name = im.split("/")[-1]
        all_poly = poly.loc[poly.file_name == im_name]['combined_labels']
        if not all_poly.empty:
            ind = all_poly.index[0]
            all_poly = all_poly.to_dict()[ind]
            if isinstance(all_poly, str):
                all_poly = ast.literal_eval(all_poly)
                filepath = os.path.join(label_dir, im_name.split(".")[0] + ".png")
                height = image.shape[0]
                width = image.shape[1]
                mask = np.zeros((height, width))
                for key, value in all_poly.items():
                    if value:
                        # change the code below to add more classes
                        for item in value:
                            cur_coords = np.array([item])
                            if key == "ORE":
                                cv2.fillPoly(mask, cur_coords, 1)
                            elif key == "CW":
                                cv2.fillPoly(mask, cur_coords, 2)
                            elif key == "DW":
                                cv2.fillPoly(mask, cur_coords, 3)
                print("Unique classes", np.unique(mask))
                cv2.imwrite(filepath, mask)
        else:
            non_labeled.append(im_name)
    print("label not created for: {} files".format(len(non_labeled)))

def make_train_test(label_dir, data_dir, train_dir, val_dir, val_label_dir, test_dir, num_test, superimpose_dir):
    """
    Getting images with labels into a separate directory

    Parameters:
    -----------
    label_dir: str
            path to the directory for saving label images
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
    # get the images and labels from the directory
    all_images = glob(os.path.join(data_dir, '*.jpg'))
    all_images.extend(glob(os.path.join(data_dir, "*.JPG")))
    labeled_images = glob(os.path.join(label_dir, '*.png'))

    train_val_images = []
    lab_images = []
    for image in labeled_images:
        lab_images.append(image.split("/")[-1].split(".")[0])

    # copy training images
    for im in all_images:
        if im.split("/")[-1].split(".")[0] in lab_images:
            train_val_images.append(im)
            if os.path.isfile(im):
                shutil.copy(im, train_dir)

    # copy test images
    i = 0
    for im in all_images:
        i += 1
        if i == num_test:
            break
        if im.split("/")[-1].split(".")[0] not in lab_images:
            shutil.copy(im, test_dir)

    # uncomment the code chunk below to superimpose the images on labels

    # if args.superimpose_dir:
    #     print("Superimposing the images and labels..")
    #     if not os.path.exists(args.superimpose_dir):
    #         os.makedirs(args.superimpose_dir)
    #     superimpose(args.train_dir, args.label_dir, args.superimpose_dir)

    # do a train-val split
    random.shuffle(train_val_images)
    print("Splitting into train and validation sets..")
    val_images = train_val_images[:args.num_valid]
    for im in val_images:
        im_name = im.split("/")[-1]
        im_path = os.path.join(train_dir, im_name)
        lab_path = os.path.join(label_dir, im_name.split(".")[0] + ".png")
        if os.path.exists(im) and os.path.exists(lab_path):
            shutil.move(im_path, val_dir)
            shutil.move(lab_path, val_label_dir)
    print("All datasets prepared!")

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
    for im in image_list:
        image_name = im.split("/")[-1]
        # image_path = os.path.join(train_dir, image_name)
        label_path = os.path.join(label_dir, image_name.split(".")[0] + ".png")
        if os.path.isfile(im) and os.path.isfile(label_path):
            image = cv2.imread(im)
            label = cv2.imread(label_path)
            label = give_color_to_seg_img(label, 4)
            label = (label + 1) * (255.0 / 2)   # rescaling the labels
            if image.shape == label.shape:
                output = cv2.addWeighted(image, 0.6, label, 0.4, 0, dtype = 0)
                cv2.imwrite(os.path.join(superimpose_dir, image_name), output)
            else:
                print(f"This image doesn't have a compatible sized label {image_name}")
        else:
            print(f"One of the image and the label doesn't exist: {im}, {label_path}")

def main():
    if not os.path.exists(args.label_dir):
        os.makedirs(args.label_dir)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if not os.path.exists(args.val_dir):
        os.makedirs(args.val_dir)
    if not os.path.exists(args.val_label_dir):
        os.makedirs(args.val_label_dir)

    print("Reading in the polygon labels..")
    poly, image_list = read_data(args.input_file, args.data_dir)
    print("Extracting and exporting label information..")
    export_labels(poly, image_list, args.data_dir, args.label_dir)
    print("Making directories with training images and labels..")
    make_train_test(args.label_dir, args.data_dir, args.train_dir, args.val_dir,
                    args.val_label_dir, args.test_dir, args.num_test, args.superimpose_dir)
    print("Process completed!")

if __name__ == "__main__":
    main()
