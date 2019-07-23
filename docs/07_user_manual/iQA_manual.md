Image Quality Assurance (iQA)

**PLEASE READ BEFORE USING THE APP**
This is an user manual of our interactive dashboard, iQA (image Quality Assurance) using [R Shiny](https://shiny.rstudio.com/). 
It lets users play around with different images to assess if pictures were taken and labeled correctly and if the predicted mask after training is accurate.

## Quickstart

First, download the source code for this application from Gitlab by typing the following code in Terminal/Git bash:

```
git clone https://gitlab.com/mfqqio/ubc_geodetection_review.git
```

After that, navigate to the iQA folder from your browser:

```
cd ironn/modules/dashboards/iQA
```

Click on R.app


When running app.R, the working directory should be set to the directory the app.R file resides in.

While this should be automatic, one should check with getwd().

This app allows a user to go through all or a specific set of images to see how well they fared in the predictive algorithm. 

The app will look at images in three dummy folders explained below:
-	The 'original' folder holds the original images
-	The 'superimposed_train_labelme' folder holds the image with the labelMe mask superimposed on it
-	The 'train_predict folder' that holds the images with the predicted mask superimposed

The logic is that the different types of images (Original unlabelled, LabelMe labelled
and the prediction label) will be stored in different folders. 
The filename for each of the 3 version of the images has to be the same. Otherwise, a regex rule has to be added.

## How to use the App
Once the app is started it will initially load all of the pictures in the 'original' folder.
If the user wants to upload a specific set of images that are present in the 'original' folder, they can upload a csv (included in the app is an example csv file 'sample_list.csv') to just run through a smaller set of images.

The header checkbox tells the app if the csv file has a header or not

What the buttons do:

-	Next: goes to the next image
-	Previous: goes to the previous image
-	Save: saves the selection of the left side for that particular picture
(clicking it multiple times will create multiple entries, so click once when satisfied with your answers).
The responses are saved in the 'responses' folder as individual csv files.
-	Download Responses: collates all of the saved csv responses into a single response
csv file(you will need to name the file and add the csv extension when saving)
-	Delete Contents: removes all of the csv files in the responses folder.
