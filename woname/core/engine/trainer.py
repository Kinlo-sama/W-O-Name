from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .configs import TrainerConfig
from woname.evaluators.registry import EVALUATORS

class Trainer:
    def __init__(
            self,
            cfg: TrainerConfig
    ):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device
        )
        self.evaluators = {}
        if cfg.evaluators is not None:
            self.evaluators = {
                evaluator_cfg.type: EVALUATORS.build(evaluator_cfg)
                for evaluator_cfg in cfg.evaluators
            }    
    def fit(
            self,
            model,
            train_loader: DataLoader,
            optimizer,
            criterion,
            val_loaders: DataLoader = None
    ):
        model.to(self.device)

        for epoch in range(self.cfg.epochs):
            train_results = self.train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                self.evaluators,
                epoch
            )
            train_loss = train_results["loss"]
            train_metrics = train_results["metrics"]
            metrics_str = " - ".join(
                f"{name}: {value:.4f}"
                for name, value in train_metrics.items()
            )
            if val_loaders is not None:
                val_loss = self.validate_one_epoch(
                    model, 
                    val_loaders,
                    criterion,
                )
                print(
                f"Epoch {epoch+1}/{self.cfg.epochs} "
                f"Train Loss: {train_loss:.4f} "
                f"TrainDice: {metrics_str['dice']:.4f}"
                f"- val_loss: {val_loss:.4f}"
            )
            else:
                print(
                    f"Epoch {epoch+1}/{self.cfg.epochs} "
                    f"Train Loss: {train_loss:.4f} "
                    f"{metrics_str}"
                )
    
    def train_one_epoch(
            self,
            model,
            loader,
            optimizer,
            criterion,
            evaluators,
            epoch
    ):
        model.train()
        running_loss = 0.0
        running_metrics = {}
        if evaluators:
            running_metrics = {
                name: 0.0
                for name in evaluators
            }
        
        pbar = tqdm(loader, desc=f"{epoch + 1}/{self.cfg.epochs + 1}", leave=False)
        for batch in pbar:

            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(
                outputs,
                targets
            )
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            postfix = {"loss": loss.item()}
            if evaluators:
                for name, evaluator in evaluators.items():
                    value = evaluator(
                        outputs,
                        targets
                    )
                    running_metrics[name] += value.item()
                    postfix[name] = running_metrics[name] / (pbar.n + 1)
            


            pbar.set_postfix(postfix)
            
        for name in running_metrics:
            running_metrics[name] /= len(loader)
        
        return {
            "loss": running_loss / len(loader),
            "metrics": running_metrics
        }
    
    def validate_one_epoch(
            self,
            model,
            loader,
            criterion,
    ):
        model.eval()
        running_loss = 0.0
        with torch.no_grad(): 
            pbar = tqdm(loader, desc="Validation", leave=False)   
            for batch in pbar:   
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)
                
                outputs = model(images)
                loss = criterion(
                    outputs,
                    targets
                )
                
                running_loss += loss.item()
                pbar.set_postfix({"val_loss": loss.item()})
        return running_loss / len(loader)