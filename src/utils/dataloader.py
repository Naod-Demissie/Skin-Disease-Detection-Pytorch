import os
import cv2
import pandas as pd
from glob import glob
from typing import Optional, Tuple, List
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from .config import *



class LocalDataset(Dataset):
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
    
 
class DermnetDataset(Dataset):
    def __init__(
            self, 
            data: Tuple[str, int], 
            transform: Optional[transforms.Compose] = None
    ) -> None:
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.data[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        label = torch.tensor(label)
        label = F.one_hot(label, num_classes=len(DERMNET_LABEL_NAME)).float()
        
        if self.transform:
            img = self.transform(img)
        return img, label



def _transforms(resize_size, crop_size):
    train_transform = transforms.Compose(
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
    val_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(resize_size[:-1]),
                transforms.CenterCrop(crop_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    
    test_transform= transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(resize_size[:-1]),
                transforms.CenterCrop(crop_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return train_transform, val_transform, test_transform
    


def prepare_local_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(f'{LOCAL_DATA_DIR}/verified_annotation_from_xml.csv')
    df['img_path'] =f'{LOCAL_DATA_DIR}/images/' + df['image_name']
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
    return train_df, val_df, test_df


def prepare_dermnet_data() -> Tuple[List, List, List]:
    label_names = os.listdir(f'{DERMNET_DATA_DIR}/train')
    train_data = []
    val_data = []
    for label in label_names:
        file_paths = glob(f'{DERMNET_DATA_DIR}/train/{label}/*')
        train_paths, val_paths = train_test_split(file_paths, test_size=0.2, random_state=42)
        sparse_label = label_names.index(label)
        train_data += [(path, sparse_label) for path in train_paths]
        val_data += [(path, sparse_label) for path in val_paths]

    test_data = []
    for label in label_names:
        file_paths = glob(f'{DERMNET_DATA_DIR}/test/{label}/*')
        sparse_label = label_names.index(label)
        test_data += [(path, sparse_label) for path in file_paths]

    return train_data, val_data, test_data


def local_dataloader(
        resize_size: Tuple[int, int, int], 
        crop_size: Tuple[int, int, int], 
        batch_size: int, 
        num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_df, val_df, test_df = prepare_local_data()
    train_transform, val_transform, test_transform = _transforms(resize_size[:-1], crop_size[:-1])

    train_ds = LocalDataset(df=train_df, transform=train_transform)
    val_ds = LocalDataset(df=val_df, transform=val_transform)
    test_ds = LocalDataset(df=test_df, transform=test_transform)

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


def dermnet_dataloader(
        resize_size: Tuple[int, int, int], 
        crop_size: Tuple[int, int, int], 
        batch_size: int, 
        num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    

    train_data, val_data, test_data = prepare_dermnet_data()
    train_transform, val_transform, test_transform = _transforms(resize_size[:-1], crop_size[:-1])


    train_ds = DermnetDataset(data=train_data, transform=train_transform)
    val_ds = DermnetDataset(data=val_data, transform=val_transform)
    test_ds = DermnetDataset(data=test_data, transform=test_transform)

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