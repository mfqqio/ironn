from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import argparse
from scipy.stats import skew, kurtosis
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
plt.style.use("ggplot")
import os
from utils import get_tup_avg, str2dict
import warnings
warnings.filterwarnings("ignore")

# Example:
# python ironn/modules/preprocessing/get_roughness.py --img_folder_path "data/images" --img_tbl_path "ironn/modules/output" --out_folder_path "ironn/modules/output"


FEATS = ["file_name","Type",
         "SkewnessBlue","KurtosisBlue","MeanPixelBlue",
         "SkewnessGreen","KurtosisGreen","MeanPixelGreen",
         "SkewnessRed","KurtosisRed","MeanPixelRed"]
COLOUR_FEATS = FEATS[2:]

class ImageColorMiner():

    def __init__(self, img_folder_path=None,
                 img_tbl_path=None,
                 out_folder_path=None):
        self.img_folder_path = img_folder_path
        self.img_tbl_path = img_tbl_path
        self.out_folder_path = out_folder_path

    def draw_img_polys(self, index=1):
        """
        Draw polygons on a given image and corresponding polygon coordinates
        selected from the image summary table

        Parameters:
        -----------
        index: int
            image row index; Default value is 1
        """
        if not isinstance(index, int):
            raise TypeError("index must be of int type")

        self.img_tbl = pd.read_csv(os.path.join(self.img_tbl_path,"img_tbl_w_labels.csv"), index_col=0)
        nrows = len(self.img_tbl)
        if index > nrows-1:
            raise ValueError("index should be less than number of rows")

        self.sample_imageid = self.img_tbl["file_name"][index]
        img = Image.open(os.path.join(self.img_folder_path,self.sample_imageid))
        self.sample_coords = str2dict(self.img_tbl["labels"][index])
        fnt = ImageFont.truetype('/Library/Fonts/Arial Bold Italic.ttf', size=52)
        draw = ImageDraw.Draw(img)
        for i in range(len(self.sample_coords.keys())):
            key = list(self.sample_coords.keys())[i]
            if not self.sample_coords[key]:
                pass
            elif len(self.sample_coords[key])==1:
                draw = ImageDraw.Draw(img, "RGBA")
                pts = self.sample_coords[key][0]
                draw.polygon(pts, fill=(100+i*20,100-i*20,100+i*20,125))
                draw.text(get_tup_avg(self.sample_coords[key][0]), key, font=fnt, fill=(255,255,255,128))
                del draw
            else:
                for element in self.sample_coords[key]:
                    draw = ImageDraw.Draw(img, "RGBA")
                    draw.polygon(element, fill=(100+i*20,100-i*20,100+i*20,125))
                    draw.text(get_tup_avg(element), key, font=fnt, fill=(255,255,255,128))
                    del draw
        if os.path.exists(os.path.join(self.out_folder_path,"img_poly_example.jpg")):
            os.remove(os.path.join(self.out_folder_path,"img_poly_example.jpg"))
            img.save(os.path.join(self.out_folder_path,"img_poly_example.jpg"))
        else:
            img.save(os.path.join(self.out_folder_path,"img_poly_example.jpg"))

    def extract_aoi(self, imgid, coords, imgsave=False):
        """
        Extract area of interest (AOI) that was defined by polygon coordinates

        Parameters:
        -----------
        imgid: str
            image name
        coords: list
            labelled polygon coordinates for each rock type
        imgsave: bool
            if imgsave == True, the output mask will be saved; Default value is False

        Returns:
        --------
        mask: numpy.ndarray (dimension: height * width)
            2D array that represents the label for an AOI
        out: numpy.ndarray (dimension: height * width * channel)
            3D arrays that represent the AOI
        """
        # input type check
        if not isinstance(imgid, str):
            raise TypeError("image must be of str type")
        if not isinstance(coords, list) and not isinstance(coords, dict):
            raise TypeError("coordinates must be of list or dict type")
        if not isinstance(imgsave, bool):
            raise TypeError("imgsave must be of bool type")

        image = cv2.imread(os.path.join(self.img_folder_path, imgid))
        height = image.shape[0]
        width = image.shape[1]

        # image size check
        if width < 2 or height < 2:
            raise ValueError("image width and height must be at least 2")
        if len(coords) < 4:
            raise ValueError("coordinates length should be at least 4")
        # image value check
        for i in range(len(coords)):
            if coords[i][1] > height or coords[i][0] > width:
                if coords[i][1] == height + 1 or coords[i][0] == width + 1:
                    coords[i] = (coords[i][0] - 1, coords[i][1] - 1)

        if isinstance(coords, list):
            mask = np.zeros((height, width))
            coords = np.array([coords])
            cv2.fillPoly(mask, coords, 1)
            mask = mask.astype(np.bool)
            out = np.zeros_like(image)
            out[mask] = image[mask]
            if imgsave:
                filename = "sample_img_mask.jpg"
                if os.path.exists(os.path.join(self.out_folder_path,filename)):
                    os.remove(os.path.join(self.out_folder_path,filename))
                    cv2.imwrite(os.path.join(self.out_folder_path,filename), out)
                else:
                    # im.save(self.out_folder_path+"img_polygons.jpg")
                    cv2.imwrite(os.path.join(self.out_folder_path,filename), out)
            return mask, out

    def get_roughness_feat(self, aoi):
        """
        Retrieve roughness features (skewness, kurtosis, average pixel) of each given aoi

        Parameters:
        -----------
        aoi: numpy.ndarray (dimension: height * width * channel)
            3D arrays that represent each subimage (i.e. each rock type)

        Returns:
        --------
        pixels: list
            a list that contains pixels on 3 channels (Blue, Green, Red)
        feats: list
            a list that contains 3 roughness features on 3 channels with the following order:
            ["SkewnessBlue","KurtosisBlue","MeanPixelBlue",
            "SkewnessGreen","KurtosisGreen","MeanPixelGreen",
             "SkewnessRed","KurtosisRed","MeanPixelRed"]
        """
        feats = []
        pixels = []
        for col in range(3): # 3 colour channels
            img_flatten = aoi[:,:,col].reshape(aoi.shape[0]*aoi.shape[1],1)
            img_nonzero = img_flatten[img_flatten.nonzero()]
            feats.append(skew(img_nonzero))
            feats.append(kurtosis(img_nonzero))
            feats.append(np.mean(img_nonzero))
            pixels.append(img_nonzero)
        return pixels, feats

    def draw_colour_dist(self, pixels):
        colour = ["b","g","r"]
        title = ["Blue Channel", "Green Channel", "Red Channel"]
        fig, ax = plt.subplots(3,1)
        fig.subplots_adjust(hspace=1)
        for i, col in enumerate(colour):
            sns.kdeplot(pixels[i], ax=ax[i], color=col)
            ax[i].set(title=title[i])
        plt.savefig(os.path.join(self.out_folder_path,"colour_dist.jpg"))

    def combine_rock_types(self, rock_name):
        """
        combine rock names into 3 categories
        """
        if rock_name in ["AMP","WSIF","GN","BS"]:
            rock_name = "CW"
        elif rock_name in ["QR", "MS","QZ"]:
            rock_name = "DW"
        else:
            rock_name = "ORE"
        return rock_name

    def create_roughness_tbl(self):
        """
        Generate roughness statistic table
        """
        img_tbl = pd.read_csv(os.path.join(self.img_tbl_path,"img_tbl_w_labels.csv"), index_col=0)
        num_cols = len(FEATS)
        feat_array = np.zeros((1,num_cols))
        # len(feat_array)
        for i in range(len(img_tbl)):
            # print(i)
            # return image array in Blue Green Red order
            img = img_tbl.iloc[i]["file_name"]
            coord_dict = str2dict(img_tbl.iloc[i]["labels"]) # convert string to dict
            # generate array that contains all the information for each subimage
            for key in coord_dict.keys():
                if len(coord_dict[key]) == 1:
                    item_list = []
                    # item_list.append(i) # image index
                    item_list.append(img) # image id
                    item_list.append(key) # rock type
                    try:
                        _, out = self.extract_aoi(img, coord_dict[key][0])
                        _, feats = self.get_roughness_feat(out)
                        for feat in feats:
                            item_list.append(feat)
                    except:
                        for j in range(9): # number of roughness features
                            item_list.append(np.nan)
                    feat_array = np.append(feat_array, [item_list], axis=0)
                else:
                    for item in coord_dict[key]:
                        item_list = []
                        # item_list.append(i) # image index
                        item_list.append(img) # image id
                        item_list.append(key) # rock type
                        try:
                            _, out = self.extract_aoi(img, item)
                            _, feats = self.get_roughness_feat(out)
                            for feat in feats:
                                item_list.append(feat)
                        except:
                            for j in range(9): # number of roughness features
                                item_list.append(np.nan)
                        feat_array = np.append(feat_array, [item_list], axis=0)
        feat_array = np.delete(feat_array, obj=0, axis=0)
        img_tbl_w_roughness = pd.DataFrame(feat_array, columns=FEATS)\
                                .dropna(axis=0)\
                                .reset_index(drop=True)
        img_tbl_w_roughness["CombinedType"] = img_tbl_w_roughness["Type"].apply(self.combine_rock_types)
        img_tbl_w_roughness.to_csv(os.path.join(self.out_folder_path,"img_tbl_w_roughness.csv"))


def main(args):
    ir = ImageColorMiner(img_folder_path=args.img_folder_path,
                   img_tbl_path=args.img_tbl_path,
                   out_folder_path=args.out_folder_path)
    ir.draw_img_polys(index=0)
    coords = ir.sample_coords["HEM"]
    imgid = ir.sample_imageid
    _, out = ir.extract_aoi(imgid=imgid,coords=coords[0],imgsave=True)
    pixels, feats = ir.get_roughness_feat(out)
    ir.draw_colour_dist(pixels)
    print("Generating Colour Feature Table...")
    ir.create_roughness_tbl()
    print("Feature Table Saved")


# call main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder_path", type=str,
                        help="input image folder")
    parser.add_argument("--img_tbl_path", type=str,
                        help="input csv folder")
    parser.add_argument("--out_folder_path", type=str,
                        help="output feature table folder")
    args = parser.parse_args()
    main(args)
