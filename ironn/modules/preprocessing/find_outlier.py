import os
import argparse
import numpy as np
import pandas as pd

# Example:
# python ironn/modules/preprocessing/find_outlier.py --input_csv_path="ironn/modules/output" --output_path="ironn/modules/output"


def main(args):
    df = pd.read_csv(os.path.join(args.input_csv_path, "img_tbl_w_roughness.csv"))
    rock_ls = list(df.Type.value_counts().index)
    tmp = create_outlier_tbl(df, rock_ls[0])
    for rock in rock_ls[1:]:
        tmp = pd.concat([tmp, create_outlier_tbl(df, rock)],axis=0)
    tmp = tmp.reset_index(drop=True)
    outlier_imgs = pd.DataFrame(tmp["file_name"].value_counts().index, columns=["file_name"])
    outlier_imgs.to_csv(os.path.join(args.output_path, "general_outlier_imgs.csv"))

def create_outlier_tbl(df, rocktype):
    """
    Create a outlier table for each rock type;
    Outlier is defined as values that are 2 standard deviation away
    from the mean of each colour feature

    Parameters:
    -----------
    df: pd.dataframe
        a table that contains colour features
    rocktype: str
        a given rock type

    Returns:
    --------
    outlier_notna: pd.dataframe
        a table that contain outliers for the given rocktype
    """
    feat = ['SkewnessBlue', 'KurtosisBlue', 'MeanPixelBlue',
            'SkewnessGreen', 'KurtosisGreen', 'MeanPixelGreen',
            'SkewnessRed', 'KurtosisRed', 'MeanPixelRed'] # features
    df = df[df.Type==rocktype].reset_index(drop=True)
    mean = np.mean(df[df.Type==rocktype][feat],axis=0).values
    sd = np.std(df[df.Type==rocktype][feat],axis=0).values
    ls_all = []
    for i in range(len(df)):
        feat_ls = []
        for j in range(len(feat)):
            if float(df.iloc[i][feat[j]]) < mean[j] - 2 * sd[j]:
                feat_ls.append(feat[j]+str("_lower"))
            elif float(df.iloc[i][feat[j]]) > mean[j] + 2 * sd[j]:
                feat_ls.append(feat[j]+str("_higher"))
            else:
                feat_ls.append(np.nan)
        if not feat_ls:
            ls_all.append(np.nan)
        else:
            ls_all.append(feat_ls)
    df["outlier_feat"] = ls_all
    outlier = df[df["outlier_feat"].notnull()]
    select_row = [i for i in range(len(outlier)) if outlier.iloc[i]["outlier_feat"].count(np.nan)!=9]
    outlier_notna = outlier.iloc[select_row]
    return outlier_notna


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv_path", type=str,
                        help="input table folder path")
    # parser.add_argument("--mode", type=str,
    #                     help="if mode=''")
    parser.add_argument("--output_path", type=str,
                        help="output folder path")
    args = parser.parse_args()
    main(args)
