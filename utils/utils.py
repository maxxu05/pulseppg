import os
import numpy as np
import torch
import random
from models.Base_Model import Base_ExpConfig
from datetime import datetime
from tqdm import tqdm
from tqdm._utils import _term_move_up
from threading import Timer
import time
from sklearn.preprocessing import label_binarize

def import_net(net_config: Base_ExpConfig):
    """
    Imports a neural network module based on the provided configuration.

    Args:
        net_config (Base_ExpConfig): Configuration containing network folder, file, and parameters.

    Returns:
        net: An instance of the neural network specified in the configuration.
    """
    net_folder = net_config.net_folder
    net_file = net_config.net_file
    # Dynamically import the network module
    net_module = __import__(f'nets.{net_folder}.{net_file}', fromlist=[''])
    # Retrieve the 'Net' class and instantiate it with parameters
    net_module_class = getattr(net_module, "Net")
    net = net_module_class(**net_config.params)

    return net

def import_model(model_config: Base_ExpConfig, 
                 train_data=None, train_labels=None, 
                 val_data=None, val_labels=None, 
                 test_data=None, test_labels=None, 
                 data_normalizer=None, data_clipping=None,
                 reload_ckpt=False, evalmodel=False, resume_on=False):
    """
    Imports a model based on the provided configuration and optionally loads checkpoint.

    Args:
        model_config (Base_ExpConfig): Configuration containing model folder, file, and parameters.
        train_data, train_labels, val_data, val_labels, test_data, test_labels: Dataset inputs and labels.
        data_normalizer: Normalization function.
        data_clipping: Clipping function.
        reload_ckpt (bool): If True, reload the model checkpoint.
        evalmodel (bool): If True, import model from evaluation folder.
        resume_on (bool): If True, resume training from previous state.

    Returns:
        model: An instance of the model specified in the configuration.
    """
    model_folder = model_config.model_folder
    model_file = model_config.model_file
    parentfolder = "eval" if evalmodel else "models"
    # Dynamically import the model module
    model_module = __import__(f'{parentfolder}.{model_folder}.{model_file}', fromlist=[''])
    # Retrieve the 'Model' class and instantiate it with parameters
    model_module_class = getattr(model_module, "Model")
    model = model_module_class(model_config, 
                               train_data=train_data, train_labels=train_labels,
                               val_data=val_data, val_labels=val_labels, 
                               test_data=test_data, test_labels=test_labels, 
                               data_normalizer=data_normalizer, data_clipping=data_clipping, 
                               resume_on=resume_on)
    
    if reload_ckpt:
        model.load("best")  # Load the best checkpoint if required

    return model

def load_data(data_config):
    """
    Loads data based on the provided configuration type (supervised or ssl).

    Args:
        data_config: Configuration containing data folder, type, annotations, and normalizer.

    Returns:
        final_out: A list containing data tensors and metadata paths.
    """
    if "supervised" in data_config.type:
        data_path = f"data/{data_config.data_folder}"
        final_out = []
        for mode in ["train", "val", "test"]:
            X = []
            # Load and concatenate feature annotations
            for X_anno in data_config.X_annotates:                                                   
                X_temp = torch.Tensor(np.load(os.path.join(data_path, f"{mode}_X_{X_anno}.npy")))
                X.append(X_temp)
            X = torch.cat(X, dim=1)
            y = np.load(os.path.join(data_path, f"{mode}_y_{data_config.y_annotate}.npy"))

            final_out.extend([X, y])

        final_out.append(None)
        final_out.append(None)

        return final_out
    
    elif data_config.type == "ssl": 
        # Prepare semi-supervised learning paths
        final_out= [os.path.join(data_config.data_folder, "train"), None,
                    os.path.join(data_config.data_folder, "val"), None,
                    os.path.join(data_config.data_folder, "test"), None]
                    
        if data_config.data_normalizer:
            final_out.append(os.path.join(data_config.data_folder, "normalizer"))
        else:
            final_out.append(None)
        final_out.append(data_config.data_clipping)

        return final_out

def printlog(line, path, type="a", updatestdout=False, dontwrite=False):
    """
    Logs a line to a file and optionally updates the standard output.

    Args:
        line (str): The line to log.
        path (str): The path to the log file.
        type (str): File open mode, default is append.
        updatestdout (bool): Whether to update stdout.
        dontwrite (bool): If True, don't write to the log file.
    """
    # Format the log line with a timestamp
    line = f"{datetime.now().strftime('%y/%m/%d %H:%M')} | " + line
    if updatestdout:
        prefix = _term_move_up() + '\r'
        tqdm.write(prefix + line)  # Update stdout with the line
    else:
        tqdm.write(line)
    if not dontwrite:
        with open(os.path.join(path, 'log.txt'), type) as file:
            file.write(line+'\n')  # Write the line to the log file

def set_seed(seed):
    """
    Set the seed for random number generators for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_dl_program(config, device_name, use_cudnn=True, deterministic=True, benchmark=True, use_tf32=False, max_threads=None):
    """
    Initializes the deep learning program with specified configurations.

    Args:
        config: Configuration object to set device.
        device_name (str or int): Name or index of the device.
        use_cudnn (bool): Whether to use cuDNN.
        deterministic (bool): Whether to use deterministic algorithms.
        benchmark (bool): Whether to use the cuDNN benchmark.
        use_tf32 (bool): Whether to allow TF32.
        max_threads (int): Maximum number of threads to use.
    """
    if max_threads is not None:
        torch.set_num_threads(max_threads)
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)

    devices.reverse()
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
    
    config.set_device(devices if len(devices) > 1 else devices[0])

class RepeatedTimer(object):
    """
    Repeatedly runs a function at specified intervals.

    Args:
        interval (float): Time interval between function calls.
        function (callable): The function to run.
        killafter (float): Time after which the timer should stop.
        args, kwargs: Arguments to pass to the function.
    """
    def __init__(self, interval, function, killafter=None, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start_time = time.time()
        self.killafter = killafter
        self.start()

    def _run(self):
        """Internal method to run the function and restart the timer."""
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        """Starts or restarts the timer."""
        if self.killafter is not None:
            if time.time() - self.start_time > self.killafter:
                self.stop()
                return

        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        """Stops the timer."""
        self._timer.cancel()
        self.is_running = False

def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model: The model to count parameters for.

    Returns:
        total_params (int): Total number of trainable parameters.
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    return None, total_params

def bootstrap_metric_confidence_interval(y_test, y_pred, metric_func, num_bootstrap_samples=2, confidence_level=0.95):
    """
    Calculates the confidence interval for a metric using bootstrapping.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels or outputs.
        metric_func (callable): Metric function to evaluate.
        num_bootstrap_samples (int): Number of bootstrap samples.
        confidence_level (float): Confidence level for the interval.

    Returns:
        lower_bound (float): Lower bound of confidence interval.
        upper_bound (float): Upper bound of confidence interval.
        bootstrapped_metrics (list): List of metric values from bootstrapped samples.
    """
    bootstrapped_metrics = []

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        indices = np.random.choice(range(len(y_test)), size=len(y_test)*100, replace=True)
        y_test_sample = y_test[indices]
        y_pred_sample = y_pred[indices]

        # Calculate the metric for the resampled data
        try:
            metric_value = metric_func(y_test_sample, y_pred_sample, average="macro")
        except:
            try:
                metric_value = metric_func(y_test_sample, y_pred_sample)
            except:
                try:
                    metric_value = metric_func(label_binarize(y_test_sample, classes=np.unique(y_test)), y_pred_sample, average="macro")
                except:
                    metric_value = metric_func(np.eye(2)[y_test_sample], y_pred_sample, average="macro")
        bootstrapped_metrics.append(metric_value)

    lower_bound = np.percentile(bootstrapped_metrics, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_metrics, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound, bootstrapped_metrics