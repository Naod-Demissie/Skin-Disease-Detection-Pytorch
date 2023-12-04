import os
import sys
import cv2
import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


sys.path.append('..')
from src.config import NUM_CLASSES, DATA_DIR

class ImageDataset(Dataset):
    def __init__(
            self, 
            df: pd.DataFrame, 
            transform: Optional[transforms.Compose] = None
    ) -> None:
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.df.loc[idx, 'img_path']
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        xmin, ymin, xmax, ymax = self.df.loc[idx, ['xmin', 'ymin', 'xmax', 'ymax']].values
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        img = img[ymin:ymax, xmin:xmax]

        if self.transform:
            img = self.transform(img)

        sparse_label = int(self.df.loc[idx, 'sparse_label'])
        sparse_label = torch.tensor(sparse_label)
        cat_label = F.one_hot(sparse_label, num_classes=NUM_CLASSES).float()
        
        return img, cat_label
    

def prepare_data(
        resize_size: Tuple[int, int, int], 
        crop_size: Tuple[int, int, int], 
        batch_size: int, 
        num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    df = pd.read_csv(f'{DATA_DIR}/verified_annotation_from_xml.csv')
    df['img_path'] =f'{DATA_DIR}/images/' + df['image_name']
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['label_name'] = df['label_name'].apply(lambda x: x.lower())
    df['sparse_label'] = df['label_name'].map({'atopic': 0, 'papular': 1,'scabies': 2})

    gs = GroupShuffleSplit(n_splits=2, train_size=.85, random_state=42)
    train_val_idx, test_idx = next(gs.split(df,groups=df.patient_id))
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    train_idx, val_idx = next(gs.split(train_val_df, groups=train_val_df.patient_id))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_ds = ImageDataset(
        df=train_df, 
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(resize_size[:-1]),
                transforms.CenterCrop(crop_size[:-1]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )
    val_ds = ImageDataset(
        df=val_df, 
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(resize_size[:-1]),
                transforms.CenterCrop(crop_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )

    test_ds = ImageDataset(
        df=test_df, 
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(resize_size[:-1]),
                transforms.CenterCrop(crop_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader