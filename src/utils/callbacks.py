import torch
from torch.utils.tensorboard import SummaryWriter
import os



class ModelCheckpoint:
    def __init__(self, save_path, monitor='val_loss', mode='min', min_delta=0.0001):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


    def __call__(self, epoch, model, optimizer, metrics):
        current_value = metrics
        if (self.mode == 'min' and (current_value-self.min_delta) < self.best_value) or \
           (self.mode == 'max' and (current_value+self.min_delta) > self.best_value):
            print(f"Model saved: {self.monitor} improved from {self.best_value} to {current_value}")
            self.best_value = current_value
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, self.save_path)



class TensorBoard:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def __call__(self, metric_name, metric_value, epoch):
        self.writer.add_scalar(metric_name, metric_value, epoch)


class EarlyStopping:
    def __init__(self, patience, monitor='val_loss', mode='min', min_delta=0.0001):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.min_delta = min_delta
        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, metrics):
        current_value = metrics
        if (self.mode == 'min' and current_value-self.min_delta < self.best_value) or \
           (self.mode == 'max' and current_value+self.min_delta > self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


