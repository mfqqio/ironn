# Importing the packages
import PIL.Image
# import glob
import os
import numpy as np
import matplotlib.pyplot as plt
#converts lists to dictionaries
from collections import Counter
#convert dict to csv
# import csv
#extra
import pandas as pd
import argparse

# Example:
# python ironn/modules/preprocessing/exif_summary.py --image_folder "data/images" --input_csv_folder "ironn/modules/output" --output_folder "ironn/modules/output"


def main(args):
    exif_summary(args.image_folder, args.input_csv_folder, args.output_folder)

def exif_summary(image_folder,input_csv_folder,output_folder):
    """
    Returns a pandas DF in a specified folder that contains a summary of specific exif data
    fir every image in the folder

    Parameters:
    ----------
    image_folder: str
        Specified directory path to folder with pictures
    input_csv_folder: str
        Specified directory path to where input csv (img_tlb_w_labels.csv)locates
    output_folder: str
        Specified directory path to where output exif summary table locates
    """
    #Exif key IDs
    #272 camera ID
    #37386 focal length
    #37378 lens aperture
    #41988 Zoom ratio 0 means zoom was not used
    #33434 exposure time

    #changed extension file into glob readable format
    # extension = "*."+ extension

    #specifyhing the csv file
    # csvfile = csvfile + ".csv"

    #changes the working directory
    # os.chdir(input_folder)

    #counts the images without the specific exif data point
    # bad_counter = 0

    # #Makes a list of all of the filenames
    # filelist = []
    # for ext in ("*.jpg","*.JPG"):
    #     for file in glob.glob(ext):
    #         filename = os.fsdecode(file)
    #         filelist.append(filename)
    #
    # #appends the filename to the first column in the data frame
    # df = pd.DataFrame(filelist, columns=['file_name'])
    df = pd.read_csv(os.path.join(input_csv_folder, "img_tbl_w_labels.csv"),index_col=0)
    #Columns for my dataFrame that will hold the exif data
    df['camera'] = 'HAS EXIF BUT NO CAMERA INFO'
    df['focal'] = 'HAS EXIF BUT NO FOCAL INFO'
    df['lens'] = 'HAS EXIF BUT NO LENS INFO'
    df['zoom_raw'] = 'HAS EXIF BUT NO ZOOM INFO'
    df['total_zoom'] = 'ZOOMER'
    df['exposure_time_raw'] = 'HAS EXIF BUT NO EXPOSURE TIME INFO'
    df['exposure_time_sec'] = 'MEOW_MIX'
    df['pixel_width'] = 'HAS EXIF BUT NO PIXEL WIDTH INFO'
    df['pixel_height'] = 'HAS EXIF BUT NO PIXEL HEIGHT INFO'
    df['number_of_megapixels'] = 'HAS EXIF BUT MEGAPIXEL'

    #Iterating over the first column and collecting the filename
    for column in df[['file_name']]:
        for index, row in df.iterrows():
            #vector that stores the file name
            dude = row["file_name"]
            #opens the image using the filename
            img = PIL.Image.open(os.path.join(image_folder, dude))
            #gets the exif data
            exif_data = img._getexif()
            if exif_data is None:
                #If there is no exif data
                row["camera"] = 'NO EXIF DATA ON IMAGE'
            elif 272 in exif_data:
                row["camera"] = exif_data[272]
            else:
                continue
            #For focal length column
            if exif_data is None:
                #If there is no focal exif data
                row["focal"] = 'NO EXIF DATA ON IMAGE'
            elif 37386 in exif_data:
                row["focal"] = exif_data[37386]
            else:
                continue
            #For lens type column
            if exif_data is None:
                #If there is no exif data
                row["lens"] = 'NO EXIF DATA ON IMAGE'
            elif 37378 in exif_data:
                row["lens"] = exif_data[37378]
            else:
                continue
           #For zoom column
            if exif_data is None:
                #If there is no exif data
                row["zoom_raw"] = 'NO EXIF DATA ON IMAGE'
            elif 41988 in exif_data:
                row["zoom_raw"] = exif_data[41988]
            else:
                continue
            #For exposure time
            if exif_data is None:
                #If there is no exif data
                row["exposure_time_raw"] = 'NO EXIF DATA ON IMAGE'
            elif 33434 in exif_data:
                row["exposure_time_raw"] = exif_data[33434]
            else:
                continue
    #Adding height and weight column info to the dataframe
    for column in df[['file_name']]:
        for index, row in df.iterrows():
           dude = row["file_name"]
           filename = os.fsdecode(dude)
           img = PIL.Image.open(os.path.join(image_folder,filename))
           width, height = img.size
           row["pixel_width"] = width
           row["pixel_height"] = height
    #checking if picture is landscape or portrait
    df['Width_to_Height_Ratio'] = df['pixel_width']/df['pixel_height']
    df.loc[df.Width_to_Height_Ratio <= 1, 'Landscape_or_Portrait'] = 'Portrat'
    df.loc[df.Width_to_Height_Ratio > 1, 'Landscape_or_Portrait'] = 'Landscape'

    #Megapixel info
    df['number_of_megapixels'] = df['pixel_width']*df['pixel_height']/1000000

    #THE DATA IS A TUPLE, no need to remove the brackets
    #testing that it shows it
    #print(df.lens[1][0])
    #print(df.lens[1][1])

    #Calculates the exposure time AND IT WORKSSSSSSSSSS
    for column in df[['file_name']]:
        for index, row in df.iterrows():
            dude = row["exposure_time_raw"]
            if dude == 'NO EXIF DATA ON IMAGE':
                row['exposure_time_sec'] = 'NaN'
            elif dude == 'HAS EXIF BUT NO EXPOSURE TIME INFO':
                row['exposure_time_sec'] = 'NaN'
            else:
                row['exposure_time_sec'] = row["exposure_time_raw"][0]/row["exposure_time_raw"][1]

    #Calculates the zoom ratio, AND IT WORKSSSSSSSSSS
    for column in df[['file_name']]:
        for index, row in df.iterrows():
            dude = row["zoom_raw"]
            if dude == 'NO EXIF DATA ON IMAGE':
                row['total_zoom'] = 'NaN'
            elif dude == 'HAS EXIF BUT NO ZOOM INFO':
                row['total_zoom'] = 'NaN'
            else:
                row['total_zoom'] = row["zoom_raw"][0]/row["zoom_raw"][1]

    # #Change the directory to the output directory for the csv file
    # os.chdir(output_folder)
    #Writing the dataframe to a file
    df.to_csv(os.path.join(output_folder, "exif_data_summary.csv"))

# call main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="input image folder path")
    parser.add_argument("--input_csv_folder", type=str,
                        help="input label table folder path")
    parser.add_argument("--output_folder", type=str,
                        help="output folder path")
    args = parser.parse_args()
    main(args)
