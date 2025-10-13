from relcon.eval.Base_Eval import Base_EvalClass
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import joblib
from relcon.utils.utils import printlog

from sklearn import metrics
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class Model(Base_EvalClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(
                self.config.evalnetparams["embed_dim"],
                self.config.evalnetparams["mlp_dim"],
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.evalnetparams["mlp_dim"],
                self.config.evalnetparams["class_num"],
            ),
        ).cuda()
        self.loss = nn.CrossEntropyLoss().cuda()

    def setup_eval(self, **kwargs):
        super().setup_eval(**kwargs)
        self.optimizer = torch.optim.Adam(
            list(self.trained_net.parameters()) + list(self.net.parameters()),
            lr=self.config.lr,
        )

    def setup_dataloader(self, X, y, train: bool) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(
            X.to(torch.float), torch.from_numpy(y).to(torch.float)
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=train,
            batch_size=self.config.batch_size,
            num_workers=torch.get_num_threads(),
        )

        return loader

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):

        self.trained_net.train(mode=train)
        self.net.train(mode=train)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            total_loss = 0

            total_probs = []
            total_preds = []
            total_trues = []
            for X, y in tqdm(
                dataloader, desc="Training" if train else "Evaluating", leave=False
            ):

                encoded = self.trained_net(X.cuda())
                logits = self.net(encoded)
                logits = logits.squeeze(1)
                y = y.long()
                loss = self.loss(logits, y.cuda())

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                with torch.no_grad():
                    total_loss += loss.item()
                    pred = torch.argmax(torch.sigmoid(logits), dim=1)

                total_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
                total_preds.append(pred.detach().cpu().numpy())
                total_trues.append(y)

            total_loss = total_loss / len(total_probs)

            total_probs = np.concatenate(total_probs)
            total_preds = np.concatenate(total_preds)
            total_trues = np.concatenate(total_trues)

        y_test = total_trues
        y_pred = total_preds
        total_probs = np.exp(total_probs) / np.sum(
            np.exp(total_probs), axis=-1, keepdims=True
        )

        # Calculate total metrics
        out_dict = {"F1": metrics.f1_score(y_true=y_test, y_pred=y_pred, average="macro")}
        # Return metrics as a dictionary
        return total_loss, out_dict

    def load(self):
        state_dict = torch.load(
            f"{self.run_dir}/checkpoint_best.pkl", map_location=self.device
        )

        print(self.trained_net.load_state_dict(state_dict["trained_net"]))
        print(self.net.load_state_dict(state_dict["net"]))

        printlog(f"Reloading {self.model_file} Model's CV", self.run_dir)

    def create_state_dict(self, epoch: int, test_loss) -> dict:
        state_dict = {"net": self.net.state_dict(),
                      "trained_net": self.trained_net.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "test_loss": test_loss,
                      "epoch": epoch}

        return state_dict