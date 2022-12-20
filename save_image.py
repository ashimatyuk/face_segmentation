import numpy as np
import os
from PIL import Image
from pathlib import Path
from torchvision.utils import draw_segmentation_masks
import torch
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from catalyst.dl import SupervisedRunner
from const import DIR_PATH, TEST_DIR, BATCH_SIZE
from model import DeepLabv3
from dataset import TestCeleb

#getting predictions
def save_img(
        TEST_DIR: Path,
        DIR_PATH: Path):
    """
    Returns saved predictions (image with mask overlapped)
    in 'DIR_PATH / predictions'.
    """
    val_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

    test_dataset = TestCeleb(
            TEST_DIR,
            transform=val_transform
        )

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

    model = DeepLabv3()

    runner = SupervisedRunner(
            input_key='features',
            output_key='scores',
            target_key='targets',
            loss_key='loss'
        )

    predictions = np.vstack(list(map(
        lambda x: x["scores"].cpu().numpy(),
        runner.predict_loader(
            loader=test_loader,
            model=model,
            resume=Path(DIR_PATH, 'model.best.pth')
            )
    )))
    transform_img = A.Compose([
        A.Resize(512, 512),
        ToTensorV2()
    ])

    test_images = os.listdir(TEST_DIR)

    for i, img in enumerate(test_images):
        test_img = np.array(
            Image.open(Path(TEST_DIR, f'{img}')).convert('RGB'))
        transformed = transform_img(image=test_img)
        test_img = transformed["image"]
        preds = draw_segmentation_masks(
            image=test_img.type(torch.uint8),
            masks=torch.gt(torch.from_numpy(predictions[i]), 0.1),
            alpha=0.7,
            colors='#8000ff'
        )
        preds = preds.detach()
        preds = F.to_pil_image(preds)
        preds.save(Path(DIR_PATH, 'predictions', f'{i}.jpg'))

save_img(TEST_DIR, DIR_PATH)