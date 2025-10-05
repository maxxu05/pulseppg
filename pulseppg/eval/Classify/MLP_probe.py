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
from sklearn.neural_network import MLPClassifier


class Model(Base_EvalClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_eval(self, **kwargs):
        super().setup_eval(**kwargs)
        for param in self.trained_net.parameters():
            param.requires_grad = False

    def fit(self):
        printlog(f"Begin Training {self.model_file}", self.run_dir)

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        num_train = self.train_data.shape[0]
        num_val = self.val_data.shape[0]

        X_valtrain = torch.concatenate((self.val_data, self.train_data))
        y_valtrain = np.concatenate((self.val_labels, self.train_labels))

        X_valtrain_temp = []
        batch_size = 128  # X_valtrain.shape[0]
        self.trained_net.eval()
        with torch.no_grad():
            for i in tqdm(range(0, X_valtrain.shape[0], batch_size)):
                X_valtrain_temp.append(
                    self.trained_net(X_valtrain[i : i + batch_size].cuda())
                    .cpu()
                    .detach()
                    .numpy()
                )
        X_valtrain = np.concatenate(X_valtrain_temp)

        estimator = MLPClassifier(hidden_layer_sizes=512,
                                  early_stopping=True,
                                  n_iter_no_change=5,
                                  validation_fraction=num_val / (num_val+num_train),
                                  random_state=42,
                                  shuffle=False)
        estimator.fit(X_valtrain, y_valtrain)

        printlog(f"Finished Training {self.model_file}", self.run_dir)

        joblib.dump(estimator, f"{self.run_dir}/checkpoint_cv_best.joblib")
        state_dict = {"trained_net": self.trained_net.state_dict()}
        torch.save(state_dict, f"{self.run_dir}/checkpoint_best.pkl")

    def load(self):
        state_dict = torch.load(
            f"{self.run_dir}/checkpoint_best.pkl", map_location=self.device
        )

        print(self.trained_net.load_state_dict(state_dict["trained_net"]))
        self.estimator = joblib.load(f"{self.run_dir}/checkpoint_cv_best.joblib")

        printlog(f"Reloading {self.model_file} Model's CV", self.run_dir)

    def test(self):
        printlog(f"Loading Best From Training", self.run_dir)
        self.load()

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        X_test = torch.Tensor(self.test_data)
        y_test = self.test_labels

        self.trained_net.eval()
        X_test_temp = []
        batch_size = 128  # X_test.shape[0]
        self.trained_net.eval()
        with torch.no_grad():
            for i in tqdm(range(0, X_test.shape[0], batch_size)):
                X_test_temp.append(
                    self.trained_net(X_test[i : i + batch_size].cuda())
                    .cpu()
                    .detach()
                    .numpy()
                )
        X_test = np.concatenate(X_test_temp)

        y_pred = self.estimator.predict(X_test)

        # # legacy code
        # d = self.estimator.decision_function(X_test)
        # total_probs = np.exp(d) / np.sum(np.exp(d), axis=-1, keepdims=True)

        # Calculate total metrics
        total_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred, average="macro")

        # Build the printout string
        printoutstring = f"F1/Test={total_f1:5f}\n"
        writer.add_scalar("F1/Test", total_f1, 0)

        # Log the metrics
        printlog(printoutstring, self.run_dir)

        # Return metrics as a dictionary
        return {
            "F1": total_f1,
        }
