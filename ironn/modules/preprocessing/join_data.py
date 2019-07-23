import pandas as pd
import numpy as np
import argparse
import os
from ast import literal_eval


# read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv_path", type=str,
                    help="input EDA dataset folder path")
parser.add_argument("--output_csv_path", type=str,
                    help="output EDA dataset folder path")
args = parser.parse_args()

# Example
# python ironn/modules/preprocessing/join_data.py --input_csv_path "ironn/modules/output/" --output_csv_path "ironn/modules/output/"

def main():
    color_summary = pd.read_csv(os.path.join(args.input_csv_path,"img_tbl_w_roughness.csv"),index_col=0)
    exif_summary = pd.read_csv(os.path.join(args.input_csv_path,"exif_data_summary.csv"),index_col=0)
    df = color_summary.set_index('file_name').join(exif_summary.set_index('file_name'))
    rm_row_idx = df[(df.camera == "NO EXIF DATA ON IMAGE") | (df.camera=="HAS EXIF BUT NO CAMERA INFO")].index
    df = df.drop(index=rm_row_idx)
    df["focal"] = df["focal"].apply(tuple2ratio)
    df["lens"] = df["lens"].apply(tuple2ratio)
    df["exposure_time_sec"] = df["exposure_time_sec"]*1e3
    df["focal_comb"] = df["focal"].apply(combine_focal)
    df["lens_comb"] = df["lens"].apply(combine_aper)
    df["zoom_comb"] = df["total_zoom"].apply(combine_zoom)
    df["et_comb"] = df["exposure_time_sec"].apply(combine_exposure_time)
    df["megapx_comb"] = df["number_of_megapixels"].apply(combine_megapx)
    df["ratio_comb"] = df["Width_to_Height_Ratio"].apply(combine_wh_ratio)
    df["pixel_size"] = [tuple([df.iloc[i]["pixel_width"],df.iloc[i]["pixel_height"]]) for i in range(len(df))]
    df = df.dropna(subset=list(df.columns[2:11]), axis=0)
    df = df.fillna("N/A")
    for col in df.columns:
        if df[col].dtype != object:
            df[col] = df[col].apply(lambda x:round(x,2))
        elif col in ["total_zoom","exposure_time_sec"]:
            df[col] = [round(float(x),2) if x != "N/A" else "N/A" for x in df[col]]
        else:
            df[col] = df[col]
    df.to_csv(os.path.join(args.output_csv_path,"complete_joined_df.csv"))

def combine_focal(focal):
    """
    Combines camera focal lens into 3 categories
    """
    if focal == 4.15:
        return "4.15 mm"
    elif focal < 4.15:
        return "Less than 4.15 mm"
    elif focal > 4.15:
        return "Greater than 4.15 mm"
    else:
        return np.nan

def combine_aper(aper):
    """
    Combines camera lens aperture into 3 categories
    """
    if aper == 2.28:
        return "2.28"
    elif aper < 2.28:
        return "Less than 2.28"
    elif aper > 2.28:
        return "Greater than 2.28"
    else:
        return np.nan

def combine_zoom(zoom):
    """
    Combines camera zoom into 3 categories
    """
    if zoom <= 1:
        return "No zoom"
    elif zoom > 1:
        return "Zoom in"
    else:
        return "N/A"

def combine_exposure_time(et):
    """
    Combines camera exposure time into 3 categories
    """
    if et <= 1:
        return "1 ms or less"
    elif et > 1:
        return "Larger than 1 ms"
    else:
        return "N/A"

def combine_megapx(megapx):
    """
    Combines camera megapx into 3 categories
    """
    if megapx <= 2:
        return "2 megapixels or less"
    elif megapx > 2 and megapx <=12:
        return "Larger than 2 and less than 12 megapixels"
    else:
        return "Larger than 12 megapixels"

def combine_wh_ratio(ratio):
    """
    Combines camera width/height ratio into 2 cateogries
    """
    if ratio == 1.33:
        return "1.33"
    else:
        return "Others"

def tuple2ratio(s):
    if "EXIF" not in s:
        tmp = literal_eval(s)
        res = round(tmp[0]/tmp[1],2)
        return res
    else:
        return np.nan

# call main function
if __name__ == "__main__":
    main()
