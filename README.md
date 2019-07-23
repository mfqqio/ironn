# ironn
An image recognition project through a partnership between the University of British Columbia Master of Data Science program and Quebec Iron Ore

<img src="imgs/00_IRONN_logo_blue.png" align="right" height="50" width="150"/>
<br>
Contributors: Jingyun Chen, Socorro Dominguez, Milos Milic, Aditya Sharma


## Project Summary

This project has been developed in order for QIO to leverage Machine Learning (ML) techniques to improve ore yields while reducing operational costs. We are providing a Python package that contains a trained neural network that categorises Ore, Dilution Waste and Contamination Waste in candidate images to 85% accuracy. We are also providing two dashboard applications that can be used in assessing image quality.

## Repo Structure

```bash
    ubc_geodetection_review/
├── data
│   ├── images
│   └── tagfiles                                             # data: images + json tags                   
└── ironn                     
    ├── modules                                              # all modules for the package
    │   ├── arch                                             # model architectures used
    │   ├── dashboards                                 
    │   │   ├── iCA                                          # image Colour Analysis
    │   │   └── iQA                                          # image Quality Assurance
    │   ├── dataloader                                       # dataloader + related files
    │   ├── output                                           # preprocessing output
    │   │   └── stat_test_result
    │   └──  preprocessing                                   # preprocessing modules
    └── tests                                                # all tests for the modules
         └── test_data
```

The following is the description of each folder in detail.

### Data
#### Images
Over 1000 pictures of the mine’s blasted face in Lac Bloom, Fermont.
Different types of blast and locations. Images are jpg or JPG format.

#### Tagfiles
Label files with coordinates of different rock types. Can be linked to the images files by name. Files come in JSON format.

### IRONN
#### Modules
##### Architecture

This folder contains 4 python scripts for two model architectures including [FCN](https://arxiv.org/abs/1411.4038) and [UNET](https://arxiv.org/abs/1505.04597).

- fcn.py

  This script contains a class named __FCN8s()__, which inherits from the [torch.nn.Module class](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) and it defines the architecture of the Fully Convolutional Networks currently being used for the model.
- vgg.py

	This script contains a class named __VGGNet()__, which inherits from the [PyTorch VGG module](https://pytorch.org/docs/stable/torchvision/models.html#id2) and is used to modify the existing VGG pre-trained model architecture into a Fully Convolutional version which can be fed into the FCN8s class.

- unet.py

  	This script contains the forward pass structure of the UNET architecture in Pytorch which inherits from the `nn.Module` class.

- unet_parts.py

	This is the script where the smaller pieces of UNET architecture are defined which are then ultimately used in `unet.py`


##### Dataloader

This folder contains one script for loading in the data for training and testing based on the [PyTorch DataLoader module](https://pytorch.org/docs/stable/data.html).

- dataset.py
	This script contains the following 4 classes, which are:

	- __CustomTransform()__ class: it adds transformations to images such as rotation and cropping, which allows doing data augmentation to increase the number of images by creating multiple images from a single image
	- __QioTrain()__ class: it inherits from the [Dataset class](https://pytorch.org/docs/stable/data.html#module-torch.utils.data) from the Pytorch utils.data module and is used for loading the training dataset.
	- __QioVal()__ class: it is used for loading the validation dataset.
	- __QioTest()__ class: it is used for loading the test dataset.

##### Output

This folder contains all spreadsheets and images generated using scripts inside preprocessing folder, which includes:

- img\_tbl\_w\_labels.csv file

	It contains the name of the image file, associated polygon coordinates for individual rock type, coordinates for combined rock types, and coordinates for each convex hull. Generated using get\_polygon\_coords.py.
- img\_tbl\_w\_roughness.csv file

	It contains the name of the image files, rock type, and associated 9 colour features (i.e. skewness, kurtosis, mean pixel on 3 channels) extracted from each rock type in every image. Generated using get\_roughness.py.
- exif\_data\_summary.csv file

	It contains the name of the image files, camera type, focal length, lens aperture, zoom, exposure time, image size, and megapixel. Generated using exif\_summary.py.
- complete\_joined\_df.csv file

	It contains all the features in both roughness and exif summary data. Generated using join\_data.py.
- general\_outlier\_imgs.csv file

	It contains the name of the image files that need to be removed in the training data. Generated using find\_outlier.py.
- stat\_test\_results folder

	It contains 6 csv files including results from anova test, pairwise test, as well as calculating mean pixel values for each rock type. Generated using stat\_test.py.
- 3 sample images

	Sample images used for eda report and presentation. Generated using get\_roughness.py.

##### Preprocessing

This folder contains all python scripts that are used to conduct colour, exif data analysis, as well as preprocess training data. It includes:

-  convexhull.py

	It contains one class named __MyJarvisWalk()__, which includes set of functions that allow the user to input a list of tuples that represent a set of coordinates and return the coordinates belonging to the convex hull of the coordinates.
- exif\_summary.py

 	It takes in a specific dictionary key that looks at specific EXIF information from an image file and returns a data frame of the images analyzed and the image feature results for each image.
- export\_poly\_train\_data.py

	This script is used to create an appropriate directory structure and create labels from the raw json files and original images. After this script is run, we have the training and the testing data created from the original images and labels.
- export\_ch\_train\_data.py

	This script works similar to the export_poly_train_data.py script except that it creates the training data for the basic model used to predict blasted faces using convex hulls.
- find\_outlier.py

	It reads image roughness data, generates an outlier table for each rock type and saves as a csv file. An outlier is defined as values that are 2 standard deviations away from the mean of each colour feature.
- get\_polygon\_coord.py

	It takes every image name, corresponding tag file to extract polygon coordinates and save the results (a table that contains: image name, coordinate dictionary and convex hull) as a csv file.
- get\_roughness.py

	It takes every image name to extract roughness statistics of each area of interest and save the results in a table as a csv file.
- join\_data.py

	It combined roughness data and EXIF data and saves the results as a csv file.
- stat\_test.py

	It allows user to conduct anova test, pairwise test and determine mean pixel value on 3 channels for each rock type based on roughness data and saves test results as csv files.
- utils.py

	It contains a set of small functions that are required for the model building process.

##### Dashboards
###### iCA

This app is a proof of concept created by [plotly dash](https://www.kdnuggets.com/2018/05/overview-dash-python-framework-plotly-dashboards.html) that helps users play around with colour features and camera effect data for each rock type and detect which images have unusual behaviours and can be further used for image quality check.

This folder contains five items including:

- app.py: it contains main application code.
- apps folder: it contains two Python scripts (camera_effect.py, colouranalyzer.py).
- assets folder: it contains the IRONN logo image.
- requirements.txt: it contains all dependencies used for creating the application.
- results folder: it is used for storing all analysis results.

###### iQA

This [shiny](https://shiny.rstudio.com/) app will allow a user to go through all or a specific set of images to see how well they fared in the predictive algorithm. The app will look at images in three dummy folders explained below:

- The “original” folder holds the original images
- The “superimposed_train_labelme” folder holds the image with the labelMe mask superimposed on it
- The “train_predict” folder that holds the images with the predicted mask superimposed

The user can upload a csv file of image names they would like to analyse and then mark the results of the images. The results of the analysis are saved as a csv file.


#### Tests
##### Test data
JSON and JPG test files that are required in the tests

#### train.py

This is the main file from where the model training is invoked and contains the training loop, the validation loop and code for saving model performance.

## Dependencies

All python packages used in the product are listed below:

Python 3.6.8

- numpy==1.15.4
- pandas==0.23.0
- Pillow==5.1.0
- opencv-python== 4.1.0.25  
- matplotlib==2.2.2
- scipy==1.1.0
- statsmodels==0.9.0
- seaborn==0.9.0
- torchvision==0.2.2
- pytorch==1.0.1
- scikit-image==0.15.0
- scikit-learn==0.21.2
