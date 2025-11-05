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

from sklearn import metrics
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error, mean_poisson_deviance


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

        estimator = Ridge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0],  # Regularization strength
            'solver': ['auto', 'cholesky', 'sparse_cg']  # Solver to use in the computational routines
        }

        grid_search = GridSearchCV(estimator=estimator, 
                            param_grid=param_grid, 
                            cv=4, 
                            scoring='neg_mean_squared_error', 
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

    def test(self, test_idx=None, dontprint=False):
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

        if test_idx is not None:
            print(f"Predicted value is {y_pred[test_idx]}")

        # Calculate the metrics
        total_mae = mean_absolute_error(y_test, y_pred)
        total_sdae = standard_deviation_of_absolute_error(y_test, y_pred)
        total_mse = mean_squared_error(y_test, y_pred)
        total_sdse = standard_deviation_of_squared_error(y_test, y_pred)
        total_r2 = r2_score(y_test, y_pred)
        total_mape = mean_absolute_percentage_error(y_test, y_pred)
        total_poisson = mean_poisson_deviance(y_test, y_pred)

        # Build the printout string
        printoutstring = f"MAE/Test={total_mae:5f}\n"
        writer.add_scalar('MAE/Test', total_mae, 0)

        printoutstring += f"SDAE/Test={total_sdae:5f}\n"
        writer.add_scalar('SDAE/Test', total_sdae, 0)

        printoutstring += f"MSE/Test={total_mse:5f}\n"
        writer.add_scalar('MSE/Test', total_mse, 0)

        printoutstring += f"SDSE/Test={total_sdse:5f}\n"
        writer.add_scalar('SDSE/Test', total_sdse, 0)

        printoutstring += f"R2/Test={total_r2:5f}\n"
        writer.add_scalar('R2/Test', total_r2, 0)

        printoutstring += f"MAPE/Test={total_mape:5f}\n"
        writer.add_scalar('MAPE/Test', total_mape, 0)

        # Log the metrics
        printlog(printoutstring, self.run_dir, dontprint=dontprint)

        # Return metrics as a dictionary
        return {
            "MAE": total_mae,
            "SDAE": total_sdae, 
            "MSE": total_mse,
            "SDSE": total_sdse,  # Include SDSE
            "R2": total_r2,
            "MAPE": total_mape,
        }

def standard_deviation_of_absolute_error(true_values, predicted_values):
    # Calculate absolute errors
    absolute_errors = np.abs(np.array(true_values) - np.array(predicted_values))
    
    # Calculate and return the standard deviation of absolute errors
    return np.std(absolute_errors)

def standard_deviation_of_squared_error(true_values, predicted_values):
    # Calculate squared errors
    squared_errors = np.square(np.array(true_values) - np.array(predicted_values))
    
    # Calculate and return the standard deviation of squared errors
    return np.std(squared_errors)