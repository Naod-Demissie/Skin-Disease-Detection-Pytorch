import os
import sys
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (accuracy_score, recall_score, 
                             precision_score, f1_score, roc_auc_score)

sys.path.append('..')
from .utils.config import *
from .utils.dataloader import prepare_data
from .utils.model import Network
from .utils.callbacks import *
from .utils.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch training argument parser')
    parser.add_argument(
        '--base_model_name', 
        type=str,
        help='The name of the base model.',
    )
    parser.add_argument(
        '--dropout', 
        type=float,
        help='The dropout rate to be used.',
    )
    parser.add_argument(
        '--output_dims', 
        type=list,
        nargs='+',
        help='A list containing the number of neuron in each layer',
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        help='The batch size.',
        default=BATCH_SIZE
    )

    parser.add_argument(
        '--lr', 
        type=float,
        help='The learning rate to be used.',
        default=LEARNING_RATE

    )
    parser.add_argument(
        '--trail_name', 
        type=str,
        help='The name of the experiment for tensorboard logging.',
    )
    parser.add_argument(
        '--task', 
        type=str,
        help='Either of `train`, `val`, `test` or `predict`.',
    )
    args = parser.parse_args()


    train_dataloader, val_dataloader, test_dataloader = prepare_data(
        resize_size=RESIZE_SIZE[args.base_model_name],
        crop_size=CROP_SIZE[args.base_model_name],
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )

    _early_stop = EarlyStopping(patience=5, monitor='val_accuracy', mode='max')
    _tensorboard = TensorBoard(log_dir=TB_LOG_DIR)
    _model_checkpoint = ModelCheckpoint(save_path=f'{CKPT_DIR}/best.pt', monitor='val_accuracy', mode='max')
    _scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    MODEL = Network(
        base_model=get_base_model(args.base_model_name),
        dropout=args.dropout,
        output_dims=args.output_dims
    )

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
            optimizer = Adam,
    )
    
    trainer.fit(train_dataloader, val_dataloader)

