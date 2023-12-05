import os
import sys
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from sklearn.metrics import (accuracy_score, recall_score, 
                             precision_score, f1_score, roc_auc_score)

# sys.path.append('..')
from utils.config import *
from utils.dataloader import prepare_dermnet_data
from utils.model import Network, ResNetwork
from utils.callbacks import *
from utils.trainer import Trainer


base_model_names = [
    'resnet18',
    'resnet50',
    'efficientnet_b0', 
    'efficientnet_v2_s', 
]


for base_model_name in base_model_names:
    trail_name = base_model_name
    train_dataloader, val_dataloader, test_dataloader = prepare_dermnet_data(
        resize_size=RESIZE_SIZE[base_model_name],
        crop_size=CROP_SIZE[base_model_name],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    MODEL = Network(
        base_model=get_base_model(base_model_name),
        dropout=None,
        output_dims=[],
        num_classes=len(DERMNET_LABEL_NAME)
    )

    _optimizer = Adam(MODEL.parameters(), lr=LEARNING_RATE)
    _scheduler = ReduceLROnPlateau(_optimizer, mode='min', patience=5, factor=0.1, verbose=True) #TODO set the monitor
    _early_stop = EarlyStopping(patience=30, monitor='val_accuracy', mode='max')
    _tensorboard = TensorBoard(log_dir=f'{TB_LOG_DIR}/pretrained/{trail_name}')
    _model_checkpoint = ModelCheckpoint(save_path=f'{CKPT_DIR}/pretrained/{trail_name}.pt', monitor='val_accuracy', mode='max')

    trainer = Trainer(
        model=MODEL,
        device=DEVICES,
        num_epoch=NUM_EPOCHS,
        metrics=[accuracy_score, recall_score, 
                    precision_score, f1_score, roc_auc_score],
        loss_fn = torch.nn.CrossEntropyLoss(),
        early_stop = _early_stop,
        model_checkpoint = _model_checkpoint,
        tensorboard = _tensorboard,
        optimizer = _optimizer,
    )
    
    trainer.fit(train_dataloader, val_dataloader)
