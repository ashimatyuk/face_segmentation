# face_segmentation
![gif (2)](https://user-images.githubusercontent.com/102593339/208042788-ee023a8d-a8ae-4ab5-9b08-14fe462d5a6b.gif)

This repository contains implementation of Unet based model and DeepLabV3 based model for face segmentation task.

# Dataset
The link to git repo with dataset: [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ). 

In [data_preprocessing.py](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/data_preprocessing.py) there were prepared 2000 images for train and validation sets each. After several experiments it was revealed that using full dataset doesn't increase model performance in this task, at least as much as it takes time to be trained, so all the results are outputs from the model trained on 2000 images for 8 epochs, batch size 4- bigger batch size couldn't have been performed by existing resources.

# Model

In this task I used DeepLabV3 with resnet-50 backbone.

# Augmentations

A.ColorJitter(), A.HorizontalFlip(), A.CoarseDropout().

# Evaluation

You can find experiments log at clear.ml [here](https://app.clear.ml/projects/e9c8ee298b4442a3b20760785f6bed63/experiments/13dde7fe9a484e6aa70f8d09b092b548/output/execution) 

Top model loss is:

![image](https://user-images.githubusercontent.com/102593339/208044203-f11e5fcc-0058-481f-849c-4e68fda76ed2.png)


Here are some inference examples (see all the ones [here](https://github.com/ashimatyuk/face_segmentation/tree/main/predictions)):

<img src='https://github.com/ashimatyuk/face_segmentation/blob/main/predictions/10.jpg' width="200" height='200' /> <img src='https://github.com/ashimatyuk/face_segmentation/blob/main/predictions/14.jpg' width="200" height='200' /> <img src='https://github.com/ashimatyuk/face_segmentation/blob/main/predictions/1.jpg' width="200" height='200' />

[train pipline](https://github.com/ashimatyuk/face_segmentation/blob/main/main.py)

[model weights](https://drive.google.com/file/d/1FpJpDiMI1hpuCXA0lnsCrxpXivjlveh6/view?usp=share_link)

[train and test dataset classes](https://github.com/ashimatyuk/face_segmentation/blob/main/dataset.py)

[constants of paths, batch size, lr](https://github.com/ashimatyuk/face_segmentation/blob/main/const.py)

## Upd:

I decided to improve model's ability to perform in case of face covering. To do that I applied A.CoarseDropout(max_holes=1, max_height=0.7, max_width=0.8, min_height=0.2, min_width=0.3, mask_fill_value=120, p=0.7)- it helps to imitate face covering cases, so model could recognize non-face elements and avoid it during segmentation. Also I preprocessed masks so they have eyglasses pixels turned off if this accessorize exists on the image. It allows model to avoid eyeglasses when making predictions (you can see the gif above).

## Reproducing the experiment

Download dataset and put it in 'dir_path' (main directory of the repository). Download weights and put model.best.pth  in 'dir_path / 'model.best.pth'. Comment runner.train() function. Then you can run main.py, wait till data is preprocessed and predictions are made.
