from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from sklearn.metrics import (accuracy_score, recall_score, 
                             precision_score, f1_score)

from utils.config import *
from utils.dataloader import dermnet_dataloader
from utils.model import Network
from utils.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from utils.trainer import Trainer


if __name__ == '__main__':
    base_model_names = [
        'resnet50',
        # 'efficientnet_b0', 
        # 'efficientnet_b3', 
        # 'efficientnet_v2_s', 
    ]

    for base_model_name in base_model_names:
        trail_name = base_model_name
        train_dataloader, val_dataloader, test_dataloader = dermnet_dataloader(
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
        _tensorboard = TensorBoard(log_dir=f'{TB_LOG_DIR}/pretraining/{trail_name}')
        _model_checkpoint = ModelCheckpoint(save_path=f'{CKPT_DIR}/pretraining/{trail_name}.pt', monitor='val_accuracy', mode='max')

        trainer = Trainer(
            model=MODEL,
            device=DEVICES,
            num_epoch=NUM_EPOCHS,
            metrics=[accuracy_score, recall_score, precision_score, f1_score],
            loss_fn = torch.nn.CrossEntropyLoss(),
            early_stop = _early_stop,
            model_checkpoint = _model_checkpoint,
            tensorboard = _tensorboard,
            optimizer = _optimizer,
            scheduler= _scheduler
        )
        
        trainer.fit(train_dataloader, val_dataloader)
