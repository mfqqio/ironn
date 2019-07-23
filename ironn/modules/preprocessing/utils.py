import os
from glob import glob
import re
import torch.nn as nn
from torch.autograd import Variable
from skimage import transform
import torch
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ast import literal_eval
import seaborn as sns

np.random.seed(1234)
torch.cuda.manual_seed_all(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tup_avg(l):
    """
    Obtain the average coordinates of a list of (x coordinate,y coordinate) pairs

    Parameters:
    -----------
    l: list
        a list of (x coordinate,y coordinate) pairs

    Returns:
    --------
    avg_coord: tuple
        a tuple that contains average coordinates

    """
    a = b = 0
    for element in l:
        a += element[0]
        b += element[1]
    avg_coord = (a/float(len(l)), b/float(len(l)))
    return avg_coord

def str2dict(string):
    """
    Convert a string to dictionary

    Parameters:
    ----------
    string: str
        input string

    Returns:
    -------
    dct: dict
        output dictionary
    """
    if not isinstance(string, str):
        raise TypeError("string must be of str type")

    dct = literal_eval(string)
    return dct

def normalize(img, mean, std):
    """
    Normalize an image with specified mean and std deviation

    Parameters:
    ----------
    img: np.array
        image in the numpy array format after reading using cv2
    mean: list
        list of means with a mean value for each channel
    std: list
        list of std with a std value for each channel

    Returns:
    -------
    img: np.array
        returns normalized image
    """

    img = img/255.0
    img[0] = (img[0] - mean[0]) / std[0]
    img[1] = (img[1] - mean[1]) / std[1]
    img[2] = (img[2] - mean[2]) / std[2]
    img = np.clip(img, 0.0, 1.0)
    return img

def get_image_paths(image_path, ignore_ids=None):
    """
    get a list of path to all training image files

    Parameters:
    ----------
    image_path: str
        path to the training images folder

    Returns:
    -------
    image_list: list
        list of .jpg image files from the training data directory
    """
    image_list = glob(os.path.join(image_path, '*.jpg'))
    image_list.extend(glob(os.path.join(image_path, '*.JPG')))
    if ignore_ids:
        image_list = [item for item in image_list if item not in ignore_ids]
    return image_list

def iou(pred, target, n_class):
    """
    get the iou metric for the model fit

    Parameters:
    ----------
    pred: np.array
        predicted array from the model
    target: np.array
        target label array from the model
    n_class: int
        number of classes in the segmentation model

    Returns:
    -------
    ious: np.float
        calculated iou for the prediction and the target
    """
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious

def pixel_acc(pred, target):
    """
    get the iou metric for the model fit

    Parameters:
    ----------
    pred: np.array
        predicted array from the model
    target: np.array
        target label array from the model

    Returns:
    -------
    pix_acc: np.float
        calculated pixel-wise accuracy for the prediction and the target
    """
    correct = (pred == target).sum()
    total = (target == target).sum()
    pix_acc = correct / total
    return pix_acc

def give_color_to_seg_img(label, n_class):
    '''
    give color to a label and convert to an image

    Parameters:
    ----------
    label: np.array
        label contaning different class numbers for different pixels
    n_class: int
        number of classes in the segmentation model
    Returns:
    -------
    seg_image: np.array (H, W, 3)
        an image in the numpy array format with different colors for different
        label classes
    '''
    # to get just one channel as all channels have the same numbers
    if len(label.shape) == 3:
        label = label[:,:,0]
    seg_img = np.zeros((label.shape[0], label.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_class)

    # change code here to use a manual color pallete
    for c in range(n_class):
        segc = (label == c)
        seg_img[:,:,0] += (segc*(colors[c][0]))
        seg_img[:,:,1] += (segc*(colors[c][1]))
        seg_img[:,:,2] += (segc*(colors[c][2]))

    return seg_img

def get_segmentation_arr(label, n_class):
    '''
    convert label image to a one hot encoded format for the model

    Parameters:
    ----------
    label: np.array
        label contaning different class numbers for different pixels
    n_class: int
        number of classes in the segmentation model
    Returns:
    -------
    seg_labels: np.array (H, W, n_class)
        a numpy array with a separate channel for each label class
    '''
    # to convert from HxWx1 to HxW
    if len(label.shape) == 3:
        label = np.squeeze(label)
    seg_labels = np.zeros((label.shape[0], label.shape[1], n_class))

    for c in range(n_class-1, -1, -1):
        seg_labels[:, :, c] = (label == c).astype(int)
    seg_labels = seg_labels.astype(int)
    # print("Before returning from get_segmentation_arr", np.unique(seg_labels))
    return seg_labels

def resize_label(image_path, label):
    """
    resize generated labels to original image size for superimposition.
    Also superimposes the label on the original image

    Parameters:
    ----------
    image_path: str
        path to the image corresponding to the label
    label: np.array
        label prediction generated and converted to image format

    Returns:
    -------
    output: np.array
        superimposed test and prediction image with classes in different colors
    """
    image = cv2.imread(image_path)
    label = transform.resize(label, image.shape)
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    plt.imsave(os.path.join("predictions", "pred_" + image_path.split("/")[-1]), label)
    label = (label + 1) * (255.0 / 2)  # denormalizing label

    # getting a weighted average of the image and the label
    output = cv2.addWeighted(image.astype('float'), 0.7, label, 0.3, 0, dtype = 0)
    return output

def make_layers(cfg, batch_norm=False):
    """
    create/combine sequential layers of the CNN in order

    Parameters:
    ----------
    cfg: list
        architecture configuration with maxpool and convolutional layer sizes
    batch_norm: bool
        whether to add batch normalization or not

    Returns:
    -------
    nn.Sequential(*layers): A sequential container
        Modules will be added to it in the order they are passed
    """
    layers = []
    in_channels = 3
    # combining all layers in the required order
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

def gen_test_output(n_class, testloader, model, test_folder):
    """
    generate inference output on unseen images

    Parameters:
    ----------
    n_class: int
        number of classes in the image
    testloader: DataLoader
        a PyTorch DataLoader object for the test set
    model: torchvision.models
        a torchvision model trained on the training data
    test_folder: str
        path to the folder with the test images

    Returns:
    -------
    test_paths[i]: str
        path to the test image in that particular inference iteration
    output: np.array
        output image (test image + prediction) that needs to be saved
    """
    # evaluation mode required for validation phase
    model = model.to(device)
    model.eval();
    all_preds = []
    all_target = []
    # no gradient required for validation phase
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images = data['image']
            # target = data['label']
            # target = np.squeeze(target.numpy())
            images = images.float()
            images = Variable(images.to(device))

            output = model(images)
            output = output.cpu()
            N, c, h, w = output.shape
            pred = np.squeeze(output.detach().cpu().numpy(), axis=0)
            pred = pred.transpose((1, 2, 0))
            pred = pred.argmax(axis=2)
            # all_preds.append(np.ravel(pred))
            # all_target.append(np.ravel(target))
            print("Before giving color", np.unique(pred))
            pred = give_color_to_seg_img(pred, n_class)

            test_paths = get_image_paths(test_folder)
            output = resize_label(test_paths[i], pred)
            yield test_paths[i], output

    # used to get the confusion matrix
    # all_preds = [item for sublist in all_preds for item in sublist]
    # all_target = [item for sublist in all_target for item in sublist]
    # all_preds = pd.Series(all_preds)
    # all_target = pd.Series(all_target)
    #
    # cf = pd.crosstab(all_target, all_preds, rownames=['True'], colnames=['Predicted'], margins=True)
    # print("Confusion matrix:", cf)

def save_inference_samples(n_class, output_dir, testloader, model, test_folder):
    """
    save generated inference output as images

    Parameters:
    ----------
    output_dir: str
        path to the output directory to save inference images
    testloader: DataLoader
        a PyTorch DataLoader object for the test set
    model: torchvision.models
        a torchvision model trained on the training data
    test_folder: str
        path to the folder with the test images

    Returns:
    -------
    None
    """
    print('Training Finished. Saving inference output to: {}'.format(output_dir))
    image_outputs = gen_test_output(n_class, testloader, model, test_folder)
    for name, image in image_outputs:
        plt.imsave(os.path.join(output_dir, name.split("/")[-1]), image)
