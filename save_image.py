import numpy as np
import os
from PIL import Image
from pathlib import Path
from torchvision.utils import draw_segmentation_masks
import torch
import torchvision.transforms.functional as F
import albumentations as A
from typing import List

def save_img(
        test_images: List[str],
        test_dir: Path,
        dir_path: Path,
        transform_img: A.core.composition.Compose,
        predictions: np.ndarray):
    """
    Returns saved predictions (image with mask overlapped)
    in 'dir_path / predictions'.
    """
    #os.mkdir(Path(dir_path, 'predictions'))

    for i, img in enumerate(test_images):
        test_img = np.array(
            Image.open(Path(test_dir, f'{img}')).convert('RGB'))
        transformed = transform_img(image=test_img)
        test_img = transformed["image"]
        preds = draw_segmentation_masks(
            image=test_img.type(torch.uint8),
            masks=torch.gt(torch.from_numpy(predictions[i]), 0.99),
            alpha=0.7,
            colors='#8000ff'
        )
        preds = preds.detach()
        preds = F.to_pil_image(preds)
        preds.save(Path(dir_path, 'predictions', f'{i}.jpg'))