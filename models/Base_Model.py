import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
import os

from data.Base_Datasets import DataConfig
from nets.base_netconfig import Base_NetConfig

class Base_ExpConfig():
    """
    Base experiment configuration class that defines parameters for model training and evaluation.

    Attributes:
        model_folder (str): Directory containing model files.
        model_file (str): Name of the model file.
        data_config (DataConfig): Configuration for data handling.
        net_config (Base_NetConfig): Network configuration parameters.
        model_config: Additional model-specific configuration.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
        batch_size (int): Batch size for data loading.
        save_epochfreq (int): Frequency of saving checkpoints.
        seed (int): Random seed for reproducibility.
        eval_configs (list): Configuration for evaluation.
        timelen: Time length parameter (default is None).
        num_threads (int): Number of threads to use (default is -1).
    """
    
    def __init__(self, 
                 # model parameters
                 model_folder: str,
                 model_file: str,
                 # data parameters
                 data_config : DataConfig, 
                 # network configuration
                 net_config: Base_NetConfig = None,
                 # papagei model config
                 model_config = None,
                 # model training parameters
                 epochs=50, lr=0.001, batch_size=16, save_epochfreq=100,
                 # experiment params
                 seed=1234,
                 eval_configs=[],
                 timelen = None,
                 num_threads=-1,
                 ):
        # Initialize parameters
        self.model_folder = model_folder
        self.model_file = model_file
        self.data_config = data_config 
        self.net_config = net_config
        self.model_config = model_config
        self.epochs = epochs
        self.lr = lr 
        self.batch_size = batch_size
        self.save_epochfreq = save_epochfreq
        self.seed = seed

        # Initialize device and input dimensions
        self.device = None
        self.input_dims = None

        self.eval_configs = eval_configs
        self.timelen = timelen

    def set_device(self, device):
        """Set the device for model training."""
        self.device = device

    def set_inputdims(self, dims):
        """Set the input dimensions for the model."""
        self.input_dims = dims

    def set_rundir(self, run_dir):
        """Set the directory for saving run outputs."""
        self.run_dir = run_dir

class BaseModelClass():
    """
    A base class for model training and evaluation, providing common methods and attributes.

    Attributes:
        config (Base_ExpConfig): Experiment configuration.
        train_data, train_labels, val_data, val_labels, test_data, test_labels: Dataset components.
        data_normalizer (bool): Flag indicating whether to perform data normalization.: Function for data normalization.
        data_clipping (bool): Flag indicating whether to perform data clipping.
        model_file (str): Model file name.
        run_dir (str): Directory for experiment outputs.
        device: Device for computation (CPU/GPU).
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        save_epochfreq (int): Frequency of saving checkpoints.
        resume_on (bool): Flag to resume training from a checkpoint.
    """
    
    def __init__(
        self,
        config: Base_ExpConfig,
        train_data=None, train_labels=None,
        val_data=None, val_labels=None,
        test_data=None, test_labels=None, 
        data_normalizer = None, data_clipping=None,
        seed=10, resume_on = False
    ): 
        from utils.utils import set_seed
        set_seed(seed)  # Set the random seed for reproducibility

        # Initialize configuration and data attributes
        self.config = config
        self.train_data, self.train_labels = train_data, train_labels
        self.val_data, self.val_labels = val_data, val_labels
        self.test_data, self.test_labels = test_data, test_labels
        self.data_normalizer = data_normalizer
        self.data_clipping = data_clipping
        
        self.model_file = config.model_file

        # Setup run directory for outputs
        self.run_dir = os.path.join("experiments/out", config.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.device = config.device
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.save_epochfreq = config.save_epochfreq
        self.resume_on = resume_on
        
        from utils.utils import import_net, printlog
        try:
            self.net = import_net(config.net_config).to(self.device)  # Import and initialize the network
        except:
            pass  # Handle cases where the network import fails

    @abstractmethod
    def setup_dataloader(self, torch.Tensor, labels: torch.Tensor, train: bool) -> torch.utils.data.DataLoader:
        """
        Abstract method to setup the data loader.

        Args:
            data (torch.Tensor): Data tensor.
            labels (torch.Tensor): Label tensor.
            train (bool): Flag to indicate training or validation mode.

        Returns:
            torch.utils.data.DataLoader: Data loader for the dataset.
        """
        ...

    @abstractmethod
    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        """
        Abstract method to run one epoch of training or validation.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader for the dataset.
            train (bool): Flag to indicate training or validation mode.
        """
        ...

    @abstractmethod
    def create_state_dict(self, epoch: int):
        """
        Abstract method to create state dictionary for saving checkpoints.

        Args:
            epoch (int): Current epoch number.
        """
        ...

    def fit(self):
        """Train the model across multiple epochs, saving the best and latest checkpoints."""
        from utils.utils import printlog

        printlog(f"Begin Training {self.model_file}", self.run_dir)

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        # Setup data loaders for training and validation
        train_loader = self.setup_dataloader(X=self.train_data, y=self.train_labels, train=True)
        val_loader = self.setup_dataloader(X=self.val_data, y=self.val_labels, train=False)   
        
        train_loss_list, val_loss_list = [], []
        best_val_loss = np.inf

        start_epoch = 0
        # Check for existing checkpoints to resume training
        if self.resume_on and os.path.exists(os.path.join(self.run_dir, "checkpoint_latest.pkl")):
            state_dict = self.load(ckpt="latest", return_state_dict=True)
            self.optimizer.load_state_dict(state_dict["optimizer"])
            start_epoch = state_dict["epoch"] + 1
            printlog(f"Resuming model from epoch {state_dict['epoch']}, with {self.epochs-start_epoch} additional epochs remaining for training", self.run_dir)
            if os.path.exists(os.path.join(self.run_dir, "checkpoint_best.pkl")):
                best_state_dict = torch.load(f'{self.run_dir}/checkpoint_best.pkl', map_location=self.device)
                best_val_loss = best_state_dict['test_loss']
            else:
                best_val_loss = state_dict['test_loss']

        # Training loop
        for epoch in tqdm(range(start_epoch, self.epochs), desc=f"{self.run_dir} fit:"):
            train_loss, train_printouts = self.run_one_epoch(train_loader, train=True)
            train_loss_list.append(train_loss)

            val_loss, val_printouts = self.run_one_epoch(val_loader, train=False)
            val_loss_list.append(val_loss)
            
            state_dict = self.create_state_dict(epoch, val_loss)
            # Save checkpoints at specified frequency
            if epoch % self.save_epochfreq == 0:
                torch.save(state_dict, f'{self.run_dir}/checkpoint_epoch{epoch}.pkl')
            if epoch == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state_dict, f'{self.run_dir}/checkpoint_best.pkl')
            torch.save(state_dict, f'{self.run_dir}/checkpoint_latest.pkl')

            # Log training and validation losses
            printoutstring = f"Epoch #{epoch}: Loss/Train={train_loss:5f} | Loss/Val={val_loss:5f}"
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)

            for key in train_printouts.keys():
                writer.add_scalar(f'{key}/Train', train_printouts[key], epoch)
                writer.add_scalar(f'{key}/Val', val_printouts[key], epoch)
                printoutstring += f'\n {key}/Train: {train_printouts[key]} | {key}/Val: {val_printouts[key]} || '
            printlog(printoutstring, self.run_dir)

    def test(self):
        """Test the model using the best checkpoint obtained during training."""
        from utils.utils import printlog

        printlog(f"Loading Best From Training", self.run_dir)
        self.load()  # Load the best model checkpoint

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        # Setup data loader for testing
        test_loader = self.setup_dataloader(X=self.test_data, y=self.test_labels, train=False)

        test_loss_list = []
        test_loss, test_printouts = self.run_one_epoch(test_loader, train=False)
        test_loss_list.append(test_loss)

        epoch = 0
        # Log test loss
        printoutstring = f"Loss/Test={test_loss:5f}"
        writer.add_scalar('Loss/Test', test_loss, epoch)

        for key in test_printouts.keys():
            writer.add_scalar(f'{key}/Test', test_printouts[key], epoch)
            printoutstring += f'\n {key}/Test: {test_printouts[key]} || '
        printlog(printoutstring, self.run_dir)

        return test_printouts
        
    def load(self, ckpt="best", return_state_dict=False):
        """
        Load a model checkpoint.

        Args:
            ckpt (str): Checkpoint to load ("best" or "latest").
            return_state_dict (bool): If True, return the state dictionary.

        Returns:
            dict: State dictionary if return_state_dict is True.
        """
        from utils.utils import printlog

        state_dict = torch.load(f'{self.run_dir}/checkpoint_{ckpt}.pkl', map_location=self.device)

        print(self.net.load_state_dict(state_dict["net"]))
        printlog(f"Reloading {self.model_file} Model's ckpt {ckpt}, which is from epoch {state_dict['epoch']}", self.run_dir)
        if return_state_dict:
            return state_dict