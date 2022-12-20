import cv2
import os
import shutil
from pathlib import Path
from const import *

def data_prep(
    IMG_DIR_TRAIN_ALL: Path,
    IMG_DIR_TRAIN: Path,
    MASK_DIR: Path,
    MASK_DIR_TRAIN: Path,
    MASK_DIR_VAL: Path,
    IMG_DIR_VAL: Path):

    """
    Function creates folders containing 2000 train and 400 validation images and masks each.
    Masks also are pre-processed in order to remove eyeglasses pixels from masks.
    """

    os.mkdir(IMG_DIR_TRAIN)
    os.mkdir(MASK_DIR_TRAIN)
    os.mkdir(IMG_DIR_VAL)
    os.mkdir(MASK_DIR_VAL)

    #moving masks from 2 chosen folders (8 and 9) to mask_dir_train and mask_dir_val folders
    for path in os.listdir(MASK_DIR):
        if os.path.isdir(Path(MASK_DIR, path)) is True and int(path) == 10:
            for mask in os.listdir(Path(MASK_DIR, path)):
                mask_path = Path(MASK_DIR, path, mask)
                if 'skin' in mask_path.stem.split('_')[-1]:
                    shutil.move(mask_path, MASK_DIR_TRAIN)
        elif os.path.isdir(Path(MASK_DIR, path)) is True and int(path) == 11:
            for mask in os.listdir(Path(MASK_DIR, path)):
                val_filename = Path(MASK_DIR, path, mask).stem.split('_')[0]
                if '.DS' not in val_filename and int(val_filename) < 22400:
                    mask_path = Path(MASK_DIR, path, mask)
                    if 'skin' in mask_path.stem.split('_')[-1]:
                        shutil.move(mask_path, MASK_DIR_VAL)

    # moving images corresponded to masks to img_dir_train and img_dir_val folders
    for img in os.listdir(Path(IMG_DIR_TRAIN_ALL)):
        img_path = Path(IMG_DIR_TRAIN_ALL, img)
        if 22000 <= int(img_path.stem) < 22400:
            shutil.move(img_path, IMG_DIR_VAL)
    for img in os.listdir(Path(IMG_DIR_TRAIN_ALL)):
        img_path = Path(IMG_DIR_TRAIN_ALL, img)
        if 20000 <= int(img_path.stem) <= 21999:
            shutil.move(img_path, IMG_DIR_TRAIN)

    #removing extra '0' from skin masks filenames
    for filename in sorted(os.listdir(MASK_DIR_TRAIN)):
        new_filename = filename.split('_')
        new_filename = new_filename[0].lstrip('0')
        os.rename(Path(MASK_DIR_TRAIN, filename),
                  Path(MASK_DIR_TRAIN, f'{new_filename}_skin.png'))
    for filename in sorted(os.listdir(MASK_DIR_VAL)):
        new_filename = filename.split('_')
        new_filename = new_filename[0].lstrip('0')
        os.rename(Path(MASK_DIR_VAL, filename),
                  Path(MASK_DIR_VAL, f'{new_filename}_skin.png'))

    #cutting eyeglasses pixels from train masks of the images where eye glasses exist
    for mask in os.listdir(Path(MASK_DIR, '10')):
        mask_name = Path(MASK_DIR, '10', mask).stem.split('_')
        if len(mask_name) == 3 and 'eyeg' in mask_name[1]+mask_name[2]:
            #removing extra '0' from eye_g masks filenames
            mask_num = mask_name[0].lstrip('0')

            #picking correspondingly image with eyeglasses and mask
            img2 = cv2.imread(str(Path(MASK_DIR, '10', f'{mask_num}_eye_g.png')))
            img1 = cv2.imread(str(Path(MASK_DIR_TRAIN, f'{mask_num}_skin.png')))

            #creating ROI
            rows, cols, channels = img2.shape
            roi = img1[0:rows, 0:cols]

            #returning mask and inverse mask
            img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            #using bitwise operation to turning off pixels with eyeglasses on the image
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
            dst = cv2.add(img1_bg, img2_fg)
            img1[0:rows, 0:cols] = dst
            cv2.imwrite(str(Path(MASK_DIR_TRAIN, f'{mask_num}_skin.png')), img1)

    #same for val masks
    for mask in os.listdir(Path(MASK_DIR, '11')):
        mask_name = Path(MASK_DIR, '11', mask).stem.split('_')
        if len(mask_name) == 3 and 'eyeg' in mask_name[1] + mask_name[2] and int(mask_name[0].lstrip('0')) < 22400:
            # removing extra '0' from eye_g masks filenames
            mask_num = mask_name[0].lstrip('0')

            # picking correspondingly image with eyeglasses and mask
            img2 = cv2.imread(str(Path(MASK_DIR, '11', f'{mask_num}_eye_g.png')))
            img1 = cv2.imread(str(Path(MASK_DIR_VAL, f'{mask_num}_skin.png')))

            # creating ROI
            rows, cols, channels = img2.shape
            roi = img1[0:rows, 0:cols]

            # returning mask and inverse mask
            img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # using bitwise operation to turning off pixels with eyeglasses on the image
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
            dst = cv2.add(img1_bg, img2_fg)
            img1[0:rows, 0:cols] = dst
            cv2.imwrite(str(Path(MASK_DIR_VAL, f'{mask_num}_skin.png')), img1)

data_prep(
        IMG_DIR_TRAIN_ALL,
        IMG_DIR_TRAIN,
        MASK_DIR,
        MASK_DIR_TRAIN,
        MASK_DIR_VAL,
        IMG_DIR_VAL)
