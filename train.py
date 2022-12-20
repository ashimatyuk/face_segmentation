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
from dataset import Celeb
from model import DeepLabv3


def main():

    with torch.no_grad():
        torch.cuda.empty_cache()

    task = Task.init(project_name='face-seg', task_name='A.CoarseDropout(p=0.7) 6k images')

    #fixing seeds to make training process determined
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    seed_everything(42)


    data_prep(
        IMG_DIR_TRAIN_ALL,
        IMG_DIR_TRAIN,
        MASK_DIR,
        MASK_DIR_TRAIN,
        MASK_DIR_VAL,
        IMG_DIR_VAL)

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
        IMG_DIR_TRAIN,
        MASK_DIR_TRAIN,
        transform=train_transform
    )
    val_dataset = Celeb(
        IMG_DIR_VAL,
        MASK_DIR_VAL,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepLabv3()
    model.to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
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


if __name__ == "__main__":
    main()