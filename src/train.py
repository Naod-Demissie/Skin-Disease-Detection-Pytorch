import os
import sys
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

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

    args = parser.parse_args()


    train_dataloader, val_dataloader, test_dataloader = prepare_data(
        resize_size=RESIZE_SIZE[args.base_model_name],
        crop_size=CROP_SIZE[args.base_model_name],
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )



    MODEL = Network(
        base_model=get_base_model(args.base_model_name),
        dropout=args.dropout,
        output_dims=args.output_dims
    )

    _optimizer = Adam(MODEL.parameters(), lr=args.lr)
    _scheduler = ReduceLROnPlateau(_optimizer, mode='min', patience=5, factor=0.1, verbose=True) #TODO set the monitor
    _early_stop = EarlyStopping(patience=5, monitor='val_accuracy', mode='max')
    _tensorboard = TensorBoard(log_dir=f'{TB_LOG_DIR}/{args.trail_name}')
    _model_checkpoint = ModelCheckpoint(save_path=f'{CKPT_DIR}/{args.trail_name}.pt', monitor='val_accuracy', mode='max')

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


# python train.py --base_model_name efficientnet_b0 --dropout 0.5 --output_dims 128 64 32 --lr 0.001 --trail_name best
