import copy
import os
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

@dataclass
class RunConfig:
    learning_rate: float = 0.01
    num_epochs: int = 200
    save_each_epoch: bool = False
    output_dir: str = "saved_models"

class GCNTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, data, run_config: RunConfig, device="cpu"):
        self.model = self.model.to(device)
        x, y, edge_index = data.x.to(device), data.y.to(device), data.edge_index.to(device)
        train_mask, val_mask = data.train_mask, data.val_mask

        optimizer = Adam(self.model.parameters(), lr=run_config.learning_rate)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        best_loss = float("inf")
        best_state = None

        for epoch in tqdm(range(run_config.num_epochs), desc="Training"):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = nn.CrossEntropyLoss()(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            #scheduler.step()

            val_loss, val_acc = self.evaluate(data, device)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                tqdm.write(
                    f"Epoch {epoch+1}/{run_config.num_epochs} - "
                    f"Train loss: {loss.item():.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                if run_config.save_each_epoch:
                    self.save(f"{run_config.output_dir}/epoch_{epoch+1}")


    def evaluate(self, data, device="cpu"):
        self.model.eval()
        x, y, edge_index = data.x.to(device), data.y.to(device), data.edge_index.to(device)
        mask = data.val_mask
        with torch.no_grad():
            out = self.model(x, edge_index)
            loss = nn.CrossEntropyLoss()(out[mask], y[mask]).item()
            pred = out.argmax(dim=1)
            correct = pred[mask] == data.y[mask]
            acc = int(correct.sum())/int(mask.sum())
        return loss, acc

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pth"))
