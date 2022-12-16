import cv2
import os
import shutil
from pathlib import Path

def data_prep(
    img_dir_train_all: Path,
    img_dir_train: Path,
    mask_dir: Path,
    mask_dir_train: Path,
    mask_dir_val: Path,
    img_dir_val: Path):

    """
    Function creates folders containing 2000 train and validation images and masks each. Masks also are pre-processed
    in order to remove eyeglasses pixels from masks.
    """

    os.mkdir(img_dir_train)
    os.mkdir(mask_dir_train)
    os.mkdir(img_dir_val)
    os.mkdir(mask_dir_val)

    #moving masks from one chosen folder to mask_dir_train and mask_dir_val folders
    for path in os.listdir(mask_dir):
        if os.path.isdir(Path(mask_dir, path)) is True and int(path) == 8:
            for mask in os.listdir(Path(mask_dir, path)):
                mask_path = Path(mask_dir, path, mask)
                if 'skin' in mask_path.stem.split('_')[-1]:
                    shutil.move(mask_path, mask_dir_train)
        elif os.path.isdir(Path(mask_dir, path)) is True and int(path) == 12:
            for mask in os.listdir(Path(mask_dir, path)):
                mask_path = Path(mask_dir, path, mask)
                if 'skin' in mask_path.stem.split('_')[-1]:
                    shutil.move(mask_path, mask_dir_val)

    # moving images corresponded to masks to img_dir_train and img_dir_val folders
    for img in os.listdir(Path(img_dir_train_all)):
        img_path = Path(img_dir_train_all, img)
        if 24000 <= int(img_path.stem) <= 25999:
            shutil.move(img_path, img_dir_val)
    for img in os.listdir(Path(img_dir_train_all)):
        img_path = Path(img_dir_train_all, img)
        if 16000 <= int(img_path.stem) <= 17999:
            shutil.move(img_path, img_dir_train)

    #removing extra '0' from skin masks filenames
    for filename in sorted(os.listdir(mask_dir_train)):
        new_filename = filename.split('_')
        new_filename = new_filename[0].lstrip('0')
        os.rename(Path(mask_dir_train, filename),
                  Path(mask_dir_train, f'{new_filename}_skin.png'))
    for filename in sorted(os.listdir(mask_dir_val)):
        new_filename = filename.split('_')
        new_filename = new_filename[0].lstrip('0')
        os.rename(Path(mask_dir_val, filename),
                  Path(mask_dir_val, f'{new_filename}_skin.png'))

    #cutting eyeglasses pixels from masks of the images where eye glasses exist
    for mask in os.listdir(Path(mask_dir, '8')):
        mask_name = Path(mask_dir, '8', mask).stem.split('_')
        if len(mask_name) == 3 and 'eyeg' in mask_name[1]+mask_name[2]:
            #removing extra '0' from eye_g masks filenames
            mask_num = mask_name[0].lstrip('0')
            #mask_num = str(mask_num).split('\\')[-1]
            #picking correspondingly image with eyeglasses and mask
            img2 = cv2.imread(str(Path(mask_dir, '8', f'{mask_num}_eye_g.png')))
            img1 = cv2.imread(str(Path(mask_dir_train, f'{mask_num}_skin.png')))

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
            cv2.imwrite(str(Path(mask_dir_train, f'{mask_num}_skin.png')), img1)

