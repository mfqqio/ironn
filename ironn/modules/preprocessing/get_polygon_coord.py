import os
import pandas as pd
import numpy as np
import json
import argparse
from utils import str2dict
import collections
# Add ConvexHull
from convexhull import MyJarvisWalk

# Example
# python ironn/modules/preprocessing/get_polygon_coord.py --image_path "data/images" --jsonfile_path "data/tagfiles" --output_path "ironn/modules/output"


def main(args):
    IMAGE_LIST = [name for name in os.listdir(args.image_path) if name.endswith(".jpg") or name.endswith(".JPG")]
    TAG_LIST = [tag.split(".")[0] for tag in os.listdir(args.jsonfile_path) if tag.endswith(".json")]
    img_tbl = pd.DataFrame({"file_name": IMAGE_LIST})
    img_tbl["labels"] = img_tbl["file_name"].apply(extract_coords, TAG_LIST=TAG_LIST)
    img_tbl["combined_labels"] = img_tbl["labels"].apply(combine_coords)
    # Add columns for ConvexHull
    img_tbl["flat_coords"] = img_tbl["combined_labels"].apply(flat_all_coords)
    img_tbl["convexhull"] = img_tbl["flat_coords"].apply(convex_column)
    img_tbl = img_tbl.dropna(axis=0).reset_index(drop=True)
    img_tbl.to_csv(os.path.join(args.output_path,"img_tbl_w_labels.csv"))

def rename_type(rock_name):
    """
    Rename rock type

    Parameters:
    -----------
    rock_name: str
        a string that represents a rock type

    Returns:
    --------
    rock_name: str
        a string that represents a renamed rock type
    """
    if rock_name == "QZ":
        rock_name = "QR"
    elif rock_name == "LIM2":
        rock_name = "IFG"
    return rock_name

def combine_type_tier1(rname):
    """
    Combine rock types into 3 categories

    Parameters:
    -----------
    rname: str
        a string that represents a rock type

    Returns:
    --------
    rname: str
        a string that represents a combined rock type
    """
    if rname in ["AMP","WSIF","GN","BS"]:
        rname = "CW"
    elif rname in ["QR", "MS"]:
        rname = "DW"
    else:
        rname = "ORE"
    return rname

def combine_coords(labels):
    """
    Combine polygon coordinates into 3 categories

    Parameters:
    -----------
    labels: dict
        a dictionary that contains polygon coordinates

    Returns:
    --------
    tmp_dict_str: str
        a string that contains a dictionary of polygon coordinates for each combined rock type
    """
    try:
        tmp_dict = {}
        for key in ['DW', 'CW', 'ORE']:
            tmp_dict[key] = []
        labels = str2dict(labels)
        for key in labels.keys():
            key_combined = combine_type_tier1(key)
            for item in labels[key]:
                tmp_dict[key_combined].append(item)
        tmp_dict_str = str(tmp_dict)
        return tmp_dict_str
    except:
        return np.nan

def extract_coords(img_name, TAG_LIST):
    """
    Extract polygon coordinates from .json file for each image

    Parameters:
    -----------
    img_name: str
        image name

    Returns:
    --------
    label_annot: dict
        labelled polygon coordinates
    """
    name = img_name.split(".")[0]
    if name in TAG_LIST:
        try:
            with open(os.path.join(args.jsonfile_path, name) + ".json", 'r') as json_file:
                coord_data = json_file.read()
                obj = json.loads(coord_data)
                label_annot = {}
                type_lst = [rename_type(x["label"].upper()) for x in obj["shapes"]]
                count_dct = collections.Counter(type_lst)
                tmp = []
                for element in obj["shapes"]:
                    key = rename_type(element["label"].upper())
                    if count_dct[key] == 1:
                        curr,lst = [],[]
                        for point in element["points"]:
                            curr.append(tuple(point))
                        curr.append(curr[0])
                        lst.append(curr)
                        label_annot[key] = lst
                    else:
                        curr = []
                        for point in element["points"]:
                            curr.append(tuple(point))
                        curr.append(curr[0])
                        if key not in label_annot.keys():
                            tmp.append(curr)
                            label_annot[key] = tmp
                        else:
                            label_annot[key].append(curr)
                return str(label_annot)
        except:
            print(name+".json")
            return np.nan

def flat_all_coords(label_annot):
    """
    Gets a dictionary with polygons and flattens all the coordinates.

    Parameters:
    ----------
    label_annot: dict
        a dictionary with polygon name and coordinates.

    Returns:
    ---------
    flat_coords: list
        list of tuples of points (x, y) that are a coordinate
        of the different polygons in a json array.
    """
    # list_of_coord = []
    flat_coords = []

    try:
        label_annot = str2dict(label_annot)
        for value in label_annot.values():
            # print("value")
            for item in value:
                for tup in item:
                    flat_coords.append(tup)
        return flat_coords

    except:
        return np.nan

def convex_column(flat_coords):
    """
    Returns the points on the convex hull in Counterclockwise order.

    Parameters:
    ----------
    flat_coords: list
        list of tuples of points (x, y).

    Returns:
    --------
    convex_hull_coords: list
        list of tuples of points that represent coordinates of convex hull
    """
    try:
        ch = MyJarvisWalk()
        convex_hull_coords = ch.convex_hull(flat_coords)
        return convex_hull_coords
    except:
        return np.nan

# call main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str,
                        help="input image folder path")
    parser.add_argument("--jsonfile_path", type=str,
                        help="input json file folder path")
    parser.add_argument("--output_path", type=str,
                        help="output folder path")
    args = parser.parse_args()
    main(args)
