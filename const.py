import pathlib
from pathlib import Path

DIR_PATH = pathlib.Path.cwd()
CELEB_DIR = Path(DIR_PATH, 'CelebAMask-HQ', 'CelebAMask-HQ')
IMG_DIR_TRAIN_ALL = Path(CELEB_DIR, 'CelebA-HQ-img')
IMG_DIR_TRAIN = Path(CELEB_DIR, 'images_train')
MASK_DIR = Path(CELEB_DIR, 'CelebAMask-HQ-mask-anno')
MASK_DIR_TRAIN = Path(CELEB_DIR, 'masks_train')

IMG_DIR_VAL = Path(CELEB_DIR, 'images_val')
MASK_DIR_VAL = Path(CELEB_DIR, 'masks_val')
GLASSES_FACES = Path(CELEB_DIR, 'glasses_faces')
GLASSES_MASKS = Path(CELEB_DIR, 'glasses_masks')

TEST_DIR = Path(DIR_PATH, 'test_path')

BATCH_SIZE = 4
LR = 0.0001