import torch
from typing import Optional
from torch.utils.data import DataLoader


from tqdm.auto import tqdm
from torch.optim import Adam

from .callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from ..config import *



_early_stop = EarlyStopping(patience=5, monitor='val_loss', mode='min')
_tensorboard = TensorBoard(log_dir=TB_LOG_DIR)
_model_checkpoint = ModelCheckpoint(save_path=CKPT_DIR, monitor='val_loss', mode='min')


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_epoch: int,
        metrics: list,
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        early_stop: Optional[EarlyStopping] = None,
        model_checkpoint: Optional[ModelCheckpoint] = None,
        tensorboard: Optional[TensorBoard] = None,
        optimizer: torch.optim = Adam,
    ) -> None:
    
        self.model = model
        self.device = device
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.num_epoch = num_epoch
        self.optimizer = optimizer(self.model.parameters(), lr=LEARNING_RATE)
        self.early_stop = early_stop
        self.model_checkpoint = model_checkpoint
        self.tensorboard = tensorboard


    def _train_step(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        metrics_values = {metric_fn.__name__: 0 for metric_fn in self.metrics}
        train_loss = 0
        
        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc='Training',
            colour='blue'
        )

        for (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, -1)

            loss = self.loss_fn(outputs, targets)
            train_loss += loss.item() 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for metric_fn in self.metrics:
                metric_name = metric_fn.__name__
                metric_value = metric_fn(targets.cpu().numpy(), predicted.cpu().numpy())
                metrics_values[metric_name] += metric_value

            progress_bar.set_description(f"Epoch {epoch+1}/{self.num_epoch}")
            progress_bar.set_postfix(train_loss=loss.item(), **metrics_values)
            progress_bar.update()

        train_loss /= len(self.train_loader)
        metric_values = {name: value/len(self.train_loader) for name, value in metric_values}
        return {'loss': train_loss, **metrics_values}


    def _validate_step(self, dataloader: DataLoader, epoch: int):
        self.model.eval()
        metrics_values = {metric_fn.__name__: 0 for metric_fn in self.metrics}
        val_loss = 0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc='Validation',
            colour='green'
        )
        with torch.no_grad():
            for (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, -1)

                loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()

                for metric_fn in self.metrics:
                    metric_name = metric_fn.__name__
                    metric_value = metric_fn(targets.cpu().numpy(), predicted.cpu().numpy())
                    metrics_values[metric_name] += metric_value

                progress_bar.set_description(f"Epoch {epoch+1}/{self.num_epoch}")
                progress_bar.set_postfix(val_loss=loss.item(), **metrics_values)
                progress_bar.update()

        val_loss /= len(self.val_loader)
        metric_values = {name: value/len(self.val_loader) for name, value in metric_values}
        return {'loss': val_loss, **metrics_values}


    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        self.model.to(self.device)
        for epoch in range(0, self.num_epoch):  
        # for epoch in range(1, self.num_epoch+1):
        # for epoch in range(epoch_resume,  self.num_epoch):

            train_logs = self._train_step(dataloader=train_loader, epoch=epoch)
            val_logs = self._validate_step(dataloader=val_loader,epoch=epoch)

            self.model_checkpoint(epoch, self.model, self.optimizer, metrics=train_logs)
            if self.tensorboard:
                for log_name, log_value in train_logs:
                    self.tensorboard(f"{log_name}/train", epoch, log_value)
                for log_name, log_value in val_logs:
                    self.tensorboard(f"{log_name}/val", epoch, log_value)
            if self.early_stop(self.early_stop.monitor):
                print("Early stopping triggered.")
                break

            # if self.scheduler.step(validation_loss)


            


