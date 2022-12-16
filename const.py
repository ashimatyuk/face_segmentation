import pathlib
from pathlib import Path

dir_path = pathlib.Path.cwd()
celeb_dir = Path(dir_path, 'CelebAMask-HQ', 'CelebAMask-HQ')
img_dir_train_all = Path(celeb_dir, 'CelebA-HQ-img')
img_dir_train = Path(celeb_dir, 'images_train')
mask_dir = Path(celeb_dir, 'CelebAMask-HQ-mask-anno')
mask_dir_train = Path(celeb_dir, 'masks_train')

img_dir_val = Path(celeb_dir, 'images_val')
mask_dir_val = Path(celeb_dir, 'masks_val')
glasses_faces = Path(celeb_dir, 'glasses_faces')
glasses_masks = Path(celeb_dir, 'glasses_masks')

test_dir = Path(dir_path, 'test_path')

batch_size = 4
lr = 0.0001