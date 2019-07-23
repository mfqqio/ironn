import numpy as np
import pandas as pd
import argparse
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# Example:
# python ironn/modules/preprocessing/stat_test.py --rtbl_path "ironn/modules/output" --result_folder "ironn/modules/output/stat_test_result"

FEATS = ["file_name","Type",
         "SkewnessBlue","KurtosisBlue","MeanPixelBlue",
         "SkewnessGreen","KurtosisGreen","MeanPixelGreen",
         "SkewnessRed","KurtosisRed","MeanPixelRed"]
COLOUR_FEATS = FEATS[2:]

class StatTest():
    def __init__(self, rtbl_path=None,
                 result_folder_path=None):
        self.rtbl_path = rtbl_path
        self.result_folder_path = result_folder_path
        self.rtbl = pd.read_csv(os.path.join(self.rtbl_path,"img_tbl_w_roughness.csv"),index_col=0)\
                      .dropna(axis=0)\
                      .reset_index(drop=True)

    def tukey_pairwise_test(self, feat, group):
        """
        Perform pairwise test on image roughness features

        Parameters:
        -----------
        feat: str
            A string that represents a colour feature
        group: str
            A string that specifies which rock type tier used in pairwise test;
            If group="Type", individual such as HEM will be used in test;
            If group="CombinedType", tier I rock type such as ORE will be used in test

        Returns:
        --------
        stat_tbl: pd.DataFrame
            A table that store the result of tukey pairwise test
        """
        if not isinstance(feat, str):
            raise TypeError("feat should be of string type")

        if feat not in COLOUR_FEATS:
            raise ValueError("feat should be chosen from 'SkewnessBlue', KurtosisBlue','MeanPixelBlue',\
            'SkewnessGreen','KurtosisGreen','MeanPixelGreen', 'SkewnessRed','KurtosisRed','MeanPixelRed'")
        tukey = pairwise_tukeyhsd(endog=self.rtbl[feat],                  # Data
                                  groups=self.rtbl[group],               # Groups
                                  alpha=0.05)                             # Significance level
        summary = tukey.summary()
        stat_tbl = pd.DataFrame(summary.data[1:], columns = summary.data[0])
        return stat_tbl

    def anova_test(self, mode, group):
        """
        Perform anova test on image roughness features

        Parameters:
        -----------
        mode: str
            A string that specifies which test is used,
            If mode="all", anova test will be performed,
            if mode="pairwise", tukey pairwise test will be performed
        group: str
            A string that specifies which rock type tier used in either anova or pairwise test.
        """
        rtbl = self.rtbl
        if mode == "all":
            groups = rtbl.groupby(group).groups
            result = {}
            if group == "Type":
                for col in COLOUR_FEATS:
                # Etract individual groups (12 groups)
                    amp = rtbl[col][groups["AMP"]]
                    hem = rtbl[col][groups["HEM"]]
                    qr = rtbl[col][groups["QR"]]
                    qrif = rtbl[col][groups["QRIF"]]
                    ifg = rtbl[col][groups["IFG"]]
                    bs = rtbl[col][groups["BS"]]
                    ms = rtbl[col][groups["MS"]]
                    qrif = rtbl[col][groups["QRIF"]]
                    sif = rtbl[col][groups["SIF"]]
                    wsif = rtbl[col][groups["WSIF"]]
                    lim1 = rtbl[col][groups["LIM1"]]
                    lim12 = rtbl[col][groups["LIM1-2"]]
                    gn = rtbl[col][groups["GN"]]
                    mag = rtbl[col][groups["MAG"]]
                    result[col] = list(stats.f_oneway(amp, hem, qr, qrif, ifg, bs, ms, qrif, sif, wsif, lim1, lim12, gn, mag))
                df = pd.DataFrame(result,index=["Test statistic","p-value"]).T
                df.to_csv(os.path.join(self.result_folder_path, "anova_test_all_"+group.lower()+".csv"))
            elif group == "CombinedType":
                for col in COLOUR_FEATS:
                    ore = rtbl[col][groups["ORE"]]
                    cw = rtbl[col][groups["CW"]]
                    dw = rtbl[col][groups["DW"]]
                    result[col] = list(stats.f_oneway(ore,cw,dw))
                df = pd.DataFrame(result,index=["Test statistic","p-value"]).T
                df.to_csv(os.path.join(self.result_folder_path,"anova_test_all_"+group.lower()+".csv"))
            else:
                raise ValueError("group can be either Type or CombinedType")
        elif mode=="pairwise":
            stat_tbl = self.tukey_pairwise_test(COLOUR_FEATS[0], group)[["group1","group2","reject"]]
            stat_tbl = stat_tbl.rename(columns={"group1": "Group1", "group2": "Group2", "reject": "SkewnessBlue"})
            feat_dict = {}
            for feat in COLOUR_FEATS[1:]:
                feat_dict[feat] = self.tukey_pairwise_test(feat, group)["reject"]
            result = pd.concat([stat_tbl, pd.DataFrame(feat_dict)],axis=1)
            # count the number of rejection for each rock pairs
            count = []
            for i in range(len(result)):
                count.append(list(result.iloc[i]).count(True))
            result["CountRejection"] = count
            result = result.sort_values(by="CountRejection",axis=0,ascending=False).reset_index(drop=True)
            result.to_csv(os.path.join(self.result_folder_path,"pairwise_test_result_"+group.lower()+".csv"))
        else:
            raise ValueError("mode can be either 'all' or 'pairwise'")

    def create_mean_pixel_tbl(self, group="CombinedType"):
        """
        Create a table that contains the average value and standard deviation for each type of rock
        group: str
            A string that specifies which rock type tier used in determining mean pixel for each feature.
        """
        df = self.rtbl
        groups = df.groupby(group).groups
        stat_dct = {}
        for key in groups.keys():
            ls = []
            for col in ["MeanPixelBlue", "MeanPixelGreen", "MeanPixelRed"]:
                ls.append(np.mean(df[col][groups[key]]))
                ls.append(np.std(df[col][groups[key]]))
            stat_dct[key] = ls
        mean_df = pd.DataFrame(stat_dct, index=["AvgMeanPixelBlue","SdMeanPixelBlue",
                              "AvgMeanPixelGreen","SdMeanPixelGreen",
                              "AvgMeanPixelRed","SdMeanPixelRed"]).T
        max_value_ind = []
        for index in range(len(mean_df)):
            max_value_ind.append(np.argmax(mean_df.iloc[index])[12:])
        mean_df["MaxValueColour"] = max_value_ind
        mean_df.to_csv(os.path.join(self.result_folder_path,"mean_pixel_tbl_"+group.lower()+".csv"))

def main(args):
    st = StatTest(rtbl_path=args.rtbl_path,
                  result_folder_path=args.result_folder)
    print("Starting Stat Test...")
    st.anova_test(mode="all",group="Type")
    st.anova_test(mode="all",group="CombinedType")
    st.anova_test(mode="pairwise",group="Type")
    st.anova_test(mode="pairwise",group="CombinedType")
    st.create_mean_pixel_tbl()
    st.create_mean_pixel_tbl(group="Type")
    print("Stat Test Results Saved")


# call main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtbl_path", type=str,
                        help="feature table folder")
    parser.add_argument("--result_folder", type=str,
                        help="stat test results folder")
    args = parser.parse_args()
    main(args)
