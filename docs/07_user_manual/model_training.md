## Model

The model architecture used for training is known as Fully Convolutional Networks. They are the based on the fact that the fully connected layers in a CNN can be converted into 1x1 convolutional layers and then more maxpooling and convolutional layers can be added to the network to upsample a low resolution image. For more information on the architecture and other details, refer to the following paper on FCNs: [Fully Convolutional Networks for Semantic Segmentation, Long et al.](https://arxiv.org/abs/1411.4038).

There are more advanced architectures and algorithms that came after FCNs, a detailed overview of which can be found here: [Review of Deep Learning Algorithms for Image Semantic Segmentation](https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57)

## Training the model

The code for the model is made up for 4 basic parts:

- **The architecture**: The model architectures used for training are present in the `arch` folder. We have tried two architectures for now: FCN8s and UNET and the code for both of them are present in the directory.
- **Dataloader**: This section includes the `dataloader` folder which contains classes to load data for training, testing and validation.
- **Preprocessing**: This module contains all the scripts required for handling and processing the data from images. The script used in the model is called `utils.py` which contains all the utility functions used in the model training and prediction.
- **train.py**: This is the main training script which contains the code for the training and the validation loop. It also contains all the command line arguments used in the training process.

Below, we describe how to use each piece of code and what needs to be changed in order to train with more classes.

### Exporting training data

In order to create appropriate training data directory structure, we need to run the [export_poly_train_data.py](https://gitlab.com/mfqqio/ubc_geodetection_review/blob/master/ironn/modules/preprocessing/export_poly_train_data.py) script. This script reads in the polygon coordinates from the file [img_tbl_w_labels.csv](https://gitlab.com/mfqqio/ubc_geodetection_review/blob/master/ironn/modules/output/img_tbl_w_labels.csv) and creates 4-5 directories depending on the command line arguments.

**Usage**:

```bash
usage: export_poly_train_data.py [-h] --input_file INPUT_FILE --data_dir
                                 DATA_DIR --train_dir TRAIN_DIR --label_dir
                                 LABEL_DIR --test_dir TEST_DIR --val_dir
                                 VAL_DIR [--superimpose_dir SUPERIMPOSE_DIR]
                                 [--num_test NUM_TEST]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        input file path for the polygon coordinates
  --data_dir DATA_DIR   input directory path for original images
  --train_dir TRAIN_DIR
                        directory path for training images
  --label_dir LABEL_DIR
                        directory path for exporting labels
  --test_dir TEST_DIR   directory path for test images
  --val_dir VAL_DIR     directory path for validation images
  --superimpose_dir SUPERIMPOSE_DIR
                        directory path for exporting images superimposed with
                        labels
  --num_test NUM_TEST   number of new unseen images for the test set
```

An example command to run the script looks like this:

```bash
python3 ubc_geodetection_review/ironn/modules/preprocessing/export_poly_train_data.py \
--input_file "ubc_geodetection_review/ironn/modules/output/img_tbl_w_labels.csv" --data_dir "04_nextgen/" \
--label_dir "training_data_all_poly/labels/" --train_dir "training_data_all_poly/training" \
--test_dir "training_data_all_poly/testing" --superimpose_dir "training_data_all_poly/superimposed/" \
--val_dir "training_data_all_poly/validation" --num_test 100
```

This script picks up images from a single directory (`data_dir`) containing all the images (both labeled and unlabeled). The `superimpose_dir` is not required for the model training process and hence, is optional to use. But it is important for sanity check to see if the labels being generated align with the original images or not.

After creating the appropriate directory structure, the next step is to upload all this data to the machine where the training will be done (for ex: an AWS EC2 instance). The current code requires all the folders (training, testing/validation and labels) to be present in the root directory of the project ([ironn/](https://gitlab.com/mfqqio/ubc_geodetection_review/tree/master/ironn)) but this can be easily changed in the dataset.py file.

### Training Model

Once the data is in the correct place, the next step is to train the model. To train the model, we use the script `train.py`. This script contains the code for the training and the validation loop and is the main script to invoke the training process. All the command line arguments are also defined in this file. More arguments can be added or the current ones be modified according to the needs.

**Usage**:  python3 train.py --help

```bash
usage: train.py [-h] --output_dir OUTPUT_DIR --root_dir ROOT_DIR
                [--model MODEL] [--use_model USE_MODEL] [--epochs EPOCHS]
                [--n_class N_CLASS] [--batch_size BATCH_SIZE] [--lr LR]
                [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                [--gamma GAMMA] [--step_size STEP_SIZE] [--validate VALIDATE]
                [--acc_out ACC_OUT] [--ignore_ids IGNORE_IDS]

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        output directory for test inference
  --root_dir ROOT_DIR   root directory for the dataset
  --model MODEL         model architecture to be used for FCN
  --use_model USE_MODEL
                        path to a saved model
  --epochs EPOCHS       num of training epochs
  --n_class N_CLASS     number of label classes
  --batch_size BATCH_SIZE
                        training batch size
  --lr LR               learning rate
  --momentum MOMENTUM   momentum for SGD
  --weight_decay WEIGHT_DECAY
                        weight decay for L2 penalty
  --gamma GAMMA         multiplicative factor of learning rate decay
  --step_size STEP_SIZE
                        decay LR by a factor of gamma every step_size epochs
  --validate VALIDATE   do inference on validation images (1) or test images
                        (0)
  --acc_out ACC_OUT     path to the directory where to save model performances
  --ignore_ids IGNORE_IDS
                        path to the file with outlier image ids
```

To train the model with the default settings and validation, use this command:

```bash
 python3 train.py --output_dir "inference/" --root_dir "." --epochs 80 --n_class 4 --batch_size 32 --lr 1e-4 --validate 1
```

To train the model with the default settings and without validation, use this command:

```bash
 python3 train.py --output_dir "inference/" --root_dir "." --epochs 80 --n_class 4 --batch_size 32 --lr 1e-4 --validate 0
```

The model is saved each time the training is done with the same name `saved_model.pth`. So in order to use the saved model to do the inference, simply use:

```bash
 python3 train.py --output_dir "inference/" --root_dir "." --n_class 4 --use_model "saved_model.pth"
```

This trains the model (if not using a saved model) and creates and saves inference images in the `inference` folder for the images in the validation folder (this folder contains images which have actual labels for checking model performance). This folder can be changed to the `testing` folder in the `train.py` file by changing `qio_test = QioVal(rootdir=args.root_dir)` to `qio_test = QioTest(rootdir=args.root_dir)` (using the dataloader for the test folder instead of validation folder) where new and unseen images can be put. When handing over the data product, we train the model on all the training images available and therefore, there will be no validation images.

### What to do to add new classes?

In order to add new classes to the model with more rock types, the only change that needs to be done is in the `export_poly_train_data.py` file (but make sure that you have the appropriate column for labels in the `image_tbl_w_labels.csv` file). The part of the code that needs to be changed is this:

```python
mask = np.zeros((height, width))
for key, value in all_poly.items():
    if value:
        for item in value:
            cur_coords = np.array([item])
            if key == "ORE":
                cv2.fillPoly(mask, cur_coords, 1)
            elif key == "CW":
                cv2.fillPoly(mask, cur_coords, 2)
            elif key == "DW":
                cv2.fillPoly(mask, cur_coords, 3)
```

In the above code, more manual mapping will need to be added for the other rock types. Once that is done, this script will export appropriate label files with number of classes = `No. of rock types` + 1 (for the background). `0` will represent the background automatically.

Once this change is done and new training data created again, the model can be trained again with the `--n_class` argument equal to the appropriate number of classes.
