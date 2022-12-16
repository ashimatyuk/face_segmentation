# face_segmentation_test_task
![](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/examples/gif.gif)

This repository contains implementation of Unet based model and DeepLabV3 based model for face segmentation task.

# Dataset
The link to git repo with dataset: [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ). 

In [data_preprocessing.py](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/data_preprocessing.py) and [data_preprocessing_part.py](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/data_preprocessing_part.py) there are two ways of data preprocessing. In first full dataset (24000 images) is preparing for training, in second- 2000 images. After several experiments it was revealed that using full dataset doesn't increase model performance in this task, at least as much as it takes time to be trained, so all the results are outputs from the model trained on 2000 images for 8 epochs, batch size 4- bigger batch size couldn't have been performed by existing resources.

# Model

In this task I used DeepLabV3 with resnet-50 backbone and Unet with resnet-34 backbone. After several experiments Unet was chosen as a final solution due to lack of considerable difference in both metrics and visualized outputs and 2-times faster training process.

# Augmentations

A.ColorJitter(), A.HorizontalFlip(), A.RGBShift(). Also there were several experiments with A.ShiftScaleRotate() with different border modes.

# Evaluation

Here are evaluation results of the model: 

Dice-score = 0,96

IoU = 0,93

BCEWithLogitsLoss = 0.053.


Here are some inference examples (see all the ones [here](https://github.com/ashimatyuk/face_segmentation_test_task/tree/master/examples)):



![image](https://user-images.githubusercontent.com/102593339/206309086-6496b997-42c8-402a-a26b-b0b8d4334136.png)
![image](https://user-images.githubusercontent.com/102593339/206308935-8567a20d-6d62-4ea1-b7f7-b9bb3f725636.png)
<img src='https://user-images.githubusercontent.com/102593339/206323745-8768d031-b7eb-4ae7-bfd6-37e893a3af4e.jpg' width="240" height='256' />

[train pipline](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/main.py)

[model weights](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/logdir/checkpoints/model.best.pth)

[experiments log-file](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/logdir/csv_logger/valid.csv)

[train and test dataset classes](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/dataset.py)

[constants of paths, batch size](https://github.com/ashimatyuk/face_segmentation_test_task/blob/master/const.py)

## Upd:

I decided to improve model's ability to perform in case of face covering. To do that I applied A.CoarseDropout(max_holes=1, max_height=200, max_width=200, min_height=50, min_width=50, mask_fill_value=0, p=0.7)- it helps to imitate face covering cases, so model could recognize non-face elements and avoid it during segmentation. For this purpose I picked up deeplabv3 after some tries with Unet that weren't quite well. [Here](https://drive.google.com/file/d/1clro2o1KCv35fIylWk-CmJE8PYQGnkm5/view?usp=share_link) you can download deeplabv3 wights after 8 epochs. Below are evaluation results:

Dice-score = 0.96

IoU = 0.94

BCEWithLogitsLoss = 0.049

