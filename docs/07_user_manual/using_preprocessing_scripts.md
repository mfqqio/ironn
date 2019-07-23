This documentation will illustrate how to use scripts inside the `preprocessing` folder other than scripts that is used for model training.

### How to use

__get\_polygon\_coords.py__

It takes every image name, corresponding tag file to extract polygon coordinates and save the results (a table that contains: image name, coordinate dictionary and convex hull) as a csv file

- Inputs:
	- images folder: data/images
	- tagfiles folder: data/tagfiles
- Outputs:
	- a csv file: ironn/modules/output/img\_tbl\_w\_labels.csv
- Usage:

```
python ironn/modules/preprocessing/get_polygon_coord.py --image_path "data/images" 
--jsonfile_path "data/tagfiles" --output_path "ironn/modules/output"

```	

__get\_roughness.py__

It takes every image name to extract roughness statistics of each area of interest and save the results in a table as a csv file along with several sample images. Sample images are used in the eda report and presentation.

- Inputs:
	- images folder: data/images
	- image table with coordinates: ironn/modules/output/img\_tbl\_w\_labels.csv
- Outputs:
	- sample image that superimposes masks over one original image: ironn/modules/output/img\_poly\_example.jpg
	- sample image that contains ONLY one rock type: ironn/modules/output/sample\_img\_mask.jpg
	- sample image that contains colour profile on 3 channels for one rock type: ironn/modules/output/colour\_dist.jpg
	- roughess table: ironn/modules/output/img\_tbl\_w\_roughness.csv
- Usage:

```
python ironn/modules/preprocessing/get_roughness.py --img_folder_path "data/images" 
--img_tbl_path "ironn/modules/output" --out_folder_path "ironn/modules/output"
```

__exif\_summary.py__

It takes in a specific dictionary key that looks at specific EXIF information from an image file and returns a data frame of the images analyzed and the image feature results for each image.

- Inputs: 
	- input csv path: ironn/modules/output/img\_tbl\_w\_labels.csv
	- input image folder path: data/images
- Outputs:
	- Output csv path: ironn/modules/output/exif\_data\_summary.csv
- Usage:
	
```
python ironn/modules/preprocessing/exif_summary.py --image_folder "data/images" 
--input_csv_folder "ironn/modules/output" --output_folder "ironn/modules/output"
```

__stat\_test.py__

It allows user to conduct anova test, pairwise test and determine mean pixel value on 3 channels for each rock type based on roughness data and saves test results as csv files.

- Inputs:
	- input csv path: ironn/modules/output/img\_tbl\_w\_roughness.csv
- Outputs:
	- anova test result for individual rock type: ironn/modules/output/stat\_test\_result/anova\_test\_all\_type.csv
	- anova test result for tier I rock types: ironn/modules/output/stat\_test\_result/anova\_test\_all\_combinedtype.csv
	- pairwise test result for individual rock type: ironn/modules/output/stat\_test\_result/pairwise\_test\_result\_type.csv
	- pairwise test result for tier I rock types: ironn/modules/output/stat\_test\_result/pairwise\_test\_result\_combinedtype.csv
	- mean pixel table for individual rock type: ironn/modules/output/stat\_test\_result/mean\_pixel\_tbl\_type.csv
	- mean pixel table for tier I rock type: ironn/modules/output/stat\_test\_result/mean\_pixel\_tbl\_combinedtype.csv
- Usage:

```
python ironn/modules/preprocessing/stat_test.py --rtbl_path "ironn/modules/output" 
--result_folder "ironn/modules/output/stat_test_result"
```

__join\_data.py__

It combined roughness data and EXIF data and saves the results as a csv file.

- Inputs:
	- image roughness data: ironn/modules/output/img\_tbl\_w\_roughness.csv
	- image EXIF data: ironn/modules/output/exif\_data\_summary.csv
- Outputs:
	- joined table:  ironn/modules/output/complete\_joined\_df.csv (which is the input of iCA)
- Usage:

```
python ironn/modules/preprocessing/join_data.py --input_csv_path "ironn/modules/output/" 
--output_csv_path "ironn/modules/output/"
```

__find\_outlier.py__

It reads image roughness data, generates an outlier table for each rock type and saves as a csv file. An outlier is defined as values that are 2 standard deviations away from the mean of each colour feature. 

- Inputs:
	- image roughness data:  ironn/modules/output/img\_tbl\_w\_roughness.csv
- Outputs:
	- image outlier table:  ironn/modules/output/general\_outlier\_imgs.csv
- Usage:

```
python ironn/modules/preprocessing/find_outlier.py --input_csv_path "ironn/modules/output" 
--output_path "ironn/modules/output"
```