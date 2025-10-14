from pulseppg.eval.Base_Eval import Base_EvalClass
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import joblib
from pulseppg.utils.utils import printlog

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score


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
        param_grid ={'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs'], 'max_iter': [1000, 10_000]}


        grid_search = GridSearchCV(estimator=estimator, 
                            param_grid=param_grid, 
                            cv=4, 
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

        d = self.grid_search.decision_function(X_test)
        total_probs = np.exp(d) / np.sum(np.exp(d), axis=-1, keepdims=True)

        # Calculate total metrics
        total_f1 = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
        total_auprc = 0 # average_precision_score(y_true=y_test, y_score=total_probs, average="macro")
        total_auroc = roc_auc_score(y_true=y_test, y_score=total_probs, average="macro", multi_class='ovo')
        total_precision = precision_score(y_true=y_test, y_pred=y_pred, average="macro")
        total_recall = recall_score(y_true=y_test, y_pred=y_pred, average="macro")
        total_acc = accuracy_score(y_true=y_test, y_pred=y_pred)

        # Build the printout string
        printoutstring = f"F1/Test={total_f1:5f}\n"
        writer.add_scalar('F1/Test', total_f1, 0)

        printoutstring += f"Acc/Test={total_acc:5f}\n"
        writer.add_scalar('Acc/Test', total_acc, 0)

        printoutstring += f"Precision/Test={total_precision:5f}\n"
        writer.add_scalar('Precision/Test', total_precision, 0)

        printoutstring += f"Recall/Test={total_recall:5f}\n"
        writer.add_scalar('Recall/Test', total_recall, 0)

        printoutstring += f"AUPRC/Test={total_auprc:5f}\n"
        writer.add_scalar('AUPRC/Test', total_auprc, 0)

        printoutstring += f"AUROC/Test={total_auroc:5f}\n"
        writer.add_scalar('AUROC/Test', total_auroc, 0)

        # Log the metrics
        printlog(printoutstring, self.run_dir)

        # Return metrics as a dictionary
        return {
            "F1": total_f1,
            "Acc": total_acc,
            "Precision": total_precision,
            "Recall": total_recall,
            "AUPRC": total_auprc,
            "AUROC": total_auroc,
        }