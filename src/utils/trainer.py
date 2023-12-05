import torch
import sklearn

from typing import Optional, List
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


from tqdm.auto import tqdm
from torch.optim import Adam

from .callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from .config import *


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_epoch: int,
        metrics: List,
        optimizer: torch.optim,
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        early_stop: Optional[EarlyStopping] = None,
        model_checkpoint: Optional[ModelCheckpoint] = None,
        tensorboard: Optional[TensorBoard] = None,
        scheduler: Optional[ReduceLROnPlateau] = None
    ) -> None:

        self.model = model
        self.device = device
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.early_stop = early_stop
        self.model_checkpoint = model_checkpoint
        self.tensorboard = tensorboard
        self.scheduler = scheduler


    def _train_step(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        running_metrics = {metric_fn.__name__.rstrip('_score'): 0 for metric_fn in self.metrics}
        running_loss = 0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc='Training',
            colour='blue'
        )

        for (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=-1)
            targets = torch.argmax(targets, dim=-1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_metrics = {}
            for metric_fn in self.metrics:
                metric_name = metric_fn.__name__.rstrip('_score')
                if metric_fn.__name__ =='accuracy_score':
                    metric_value = metric_fn(
                        targets.cpu().numpy(),
                        predicted.cpu().numpy()
                    )
                else:
                    metric_value = metric_fn(
                        targets.cpu().numpy(),
                        predicted.cpu().numpy(),
                        average='macro',
                        zero_division=0
                    )
                current_metrics[metric_name] = round(metric_value, 4)
                running_metrics[metric_name] += metric_value

            progress_bar.set_description(f"Epoch {epoch+1}/{self.num_epoch}")
            progress_bar.set_postfix(loss=loss.item(), **current_metrics)
            progress_bar.update()

        running_loss /= len(dataloader)
        running_metrics = {name: value/len(dataloader) for name, value in running_metrics.items()}
        return {'loss': running_loss, **running_metrics}


    def _validate_step(self, dataloader: DataLoader, epoch: int):
        self.model.eval()
        running_metrics = {'val_' + metric_fn.__name__.rstrip('_score'): 0 for metric_fn in self.metrics}
        val_loss = 0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc='Validation',
            colour='green',
        )
        with torch.no_grad():
            for (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=-1)
                targets = torch.argmax(targets, dim=-1)

                current_metrics = {}
                for metric_fn in self.metrics:
                    metric_name = 'val_' + metric_fn.__name__.rstrip('_score')
                    if metric_fn.__name__ =='accuracy_score':
                        metric_value = metric_fn(
                            targets.cpu().numpy(),
                            predicted.cpu().numpy()
                        )
                    else:
                        metric_value = metric_fn(
                            targets.cpu().numpy(),
                            predicted.cpu().numpy(),
                            average='macro',
                            zero_division=0
                        )

                    current_metrics[metric_name] = round(metric_value, 4)
                    running_metrics[metric_name] += metric_value

                progress_bar.set_postfix(val_loss=loss.item(), **current_metrics)
                progress_bar.update()

        val_loss /= len(dataloader)
        running_metrics = {name: value/len(dataloader) for name, value in running_metrics.items()}
        return {'val_loss': val_loss, **running_metrics}


    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        self.model.to(self.device)
        for epoch in range(self.num_epoch):
            train_logs = self._train_step(dataloader=train_loader, epoch=epoch)
            val_logs = self._validate_step(dataloader=val_loader,epoch=epoch)

            checkpoint_monitor = self.model_checkpoint.monitor
            self.model_checkpoint(
                epoch, 
                self.model, 
                self.optimizer, 
                metrics=val_logs[checkpoint_monitor],
                state_dict={
                    'optimizer': self.optimizer.state_dict(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                    'current_epoch': epoch,
                    **train_logs,
                    **val_logs
                }
            )

            self.scheduler.step(metrics=val_logs[checkpoint_monitor], epoch=epoch) if self.scheduler else None

            if self.tensorboard:
                for log_name, log_value in train_logs.items():
                    self.tensorboard(log_name, log_value, epoch)
                for log_name, log_value in val_logs.items():
                    self.tensorboard(log_name, log_value, epoch)
                self.tensorboard('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            earlystop_monitor = self.early_stop.monitor
            if self.early_stop(val_logs[earlystop_monitor]):
                print("Early stopping triggered.")
                del self.model
                break



            


