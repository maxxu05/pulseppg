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

    def setup_eval(self, **kwargs):
        super().setup_eval(**kwargs)
        for param in self.trained_net.parameters():
            param.requires_grad = False

    def fit(self):
        printlog(f"Begin Training {self.model_file}", self.run_dir)

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        num_train = self.train_data.shape[0]
        num_val = self.val_data.shape[0]
        ps = PredefinedSplit(test_fold=[-1 for _ in range(num_train)] + [0 for _ in range(num_val)])

        X_trainval = torch.concatenate((self.train_data, self.val_data))
        y_trainval = np.concatenate((self.train_labels, self.val_labels))

        X_trainval_temp = []
        batch_size = 128  # X_trainval.shape[0]
        self.trained_net.eval()
        with torch.no_grad():
            for i in tqdm(range(0, X_trainval.shape[0], batch_size)):
                X_trainval_temp.append(
                    self.trained_net(X_trainval[i : i + batch_size].cuda())
                    .cpu()
                    .detach()
                    .numpy()
                )
        X_trainval = np.concatenate(X_trainval_temp)

        scaler = StandardScaler()
        X_trainval = scaler.fit_transform(X_trainval)

        estimator = LogisticRegression(random_state=42)
        param_grid = [
            {'penalty': ['l2'], 'C': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
            {'penalty': [None]}  # No regularization
        ]

        grid_search = GridSearchCV(estimator=estimator, 
                            param_grid=param_grid, 
                            cv=ps, 
                            scoring='f1_macro', 
                            verbose=self.config.verbose, 
                            n_jobs=self.config.num_threads)
        grid_search.fit(X_trainval, y_trainval)

        printlog(f"Finished Training {self.model_file}", self.run_dir)

        joblib.dump(grid_search, f"{self.run_dir}/checkpoint_cv_best.joblib")
        joblib.dump(scaler, f"{self.run_dir}/checkpoint_scaler_best.joblib")
        state_dict = {"trained_net": self.trained_net.state_dict()}
        torch.save(state_dict, f"{self.run_dir}/checkpoint_best.pkl")

    def load(self):
        state_dict = torch.load(
            f"{self.run_dir}/checkpoint_best.pkl", map_location=self.device
        )

        print(self.trained_net.load_state_dict(state_dict["trained_net"]))
        self.grid_search = joblib.load(f"{self.run_dir}/checkpoint_cv_best.joblib")
        self.scaler = joblib.load(f"{self.run_dir}/checkpoint_scaler_best.joblib")

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

        X_test = self.scaler.transform(X_test)
        y_pred = self.grid_search.predict(X_test)

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
