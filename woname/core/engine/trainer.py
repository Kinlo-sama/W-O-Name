from typing import Dict

import torch
from torch.utils.data import DataLoader

from .configs import TrainerConfig

class Trainer:
    def __init__(
            self,
            cfg: TrainerConfig
    ):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device
        )
    
    def fit(
            self,
            model,
            train_loader: DataLoader,
            optimizer,
            criterion,
            metrics: Dict = None,
            val_loaders: DataLoader = None
    ):
        model.to(self.device)
        for epoch in range(self.cfg.epochs):
            trainer_loss = self.train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                metrics
            )
            if val_loaders is not None:
                val_loss = self.validate(
                    model, 
                    val_loaders,
                    criterion,
                    metrics
                )
            print(f"Epoch {epoch+1}/{self.cfg.epochs}")
    
    def train_one_epoch(
            self,
            model,
            loader,
            optimizer,
            criterion,
            metrics
    ):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader):
            images, targets = batch
            
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(
                logits,
                targets
            )
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if metrics is not None:
                pass
        
        return running_loss / len(loader)
    
    def validate(
            self,
            model,
            loader,
            criterion,
            metrics
    ):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():    
            for step, batch in enumerate(loader):   
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                logits = model(images)
                loss = criterion(
                    logits,
                    targets
                )
                
                running_loss += loss.item()
        return running_loss / len(loader)