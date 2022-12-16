from const import *

from clearml import Task

from pathlib import Path
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from catalyst import dl
from catalyst.dl import SupervisedRunner

from data_preprocessing import data_prep
from dataset import Celeb, TestCeleb
from model import DeepLabv3
from save_image import save_img

def main():

    with torch.no_grad():
        torch.cuda.empty_cache()

    task = Task.init(project_name='face-seg', task_name='A.CoarseDropout(p=0.7)')

    #fixing seeds to make training process determined
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    seed_everything(42)


    data_prep(
        img_dir_train_all,
        img_dir_train,
        mask_dir,
        mask_dir_train,
        mask_dir_val,
        img_dir_val)

    train_transform = A.Compose([
        A.Resize(512, 512),
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        ),
        A.HorizontalFlip(),
        A.CoarseDropout(
            max_holes=1,
            max_height=0.7,
            max_width=0.8,
            min_height=0.2,
            min_width=0.3,
            fill_value=120,
            mask_fill_value=0,
            p=0.7
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

    train_dataset = Celeb(
        img_dir_train,
        mask_dir_train,
        transform=train_transform
    )
    val_dataset = Celeb(
        img_dir_val,
        mask_dir_val,
        transform=val_transform
    )
    test_dataset = TestCeleb(
        test_dir,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepLabv3()
    model.to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=1
    )

    #defining Catalyst runner parameters and running training
    runner = SupervisedRunner(
        input_key='features',
        output_key='scores',
        target_key='targets',
        loss_key='loss'
    )

    loaders = {
        "train": train_loader,
        "valid": val_loader,
    }

    runner.train(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=8,
        scheduler=scheduler,
        callbacks=[
            dl.IOUCallback(
                input_key="scores",
                target_key="targets",
                threshold=0.9
            ),
            dl.DiceCallback(
                input_key="scores",
                target_key="targets",
                threshold=0.9
            )
        ],
        logdir="./logdir",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )

    #getting predictions
    predictions = np.vstack(list(map(
        lambda x: x["scores"].cpu().numpy(),
        runner.predict_loader(
            loader=test_loader,
            model=model,
            resume=Path(dir_path, 'model.best.pth')
            )
    )))
    transform_img = A.Compose([
        A.Resize(512, 512),
        ToTensorV2()
    ])

    test_images = os.listdir(test_dir)

    save_img(test_images, test_dir, dir_path, transform_img, predictions)

if __name__ == "__main__":
    main()