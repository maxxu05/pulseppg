from models.Base_Model import Base_ExpConfig, BaseModelClass
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from tqdm import tqdm
from models.utils.CustomDatasets import OnTheFly_FolderNpyDataset

class crossmodalrebardist_ExpConfig(Base_ExpConfig):
    """
    Configuration class for the Crossmodal RebarDist experiment, extending Base_ExpConfig.

    Attributes:
        mask_extended (int): Length of the mask to be applied during training.
        mask_transient_size (int): Size of transient masks.
        mask_transient_perc (float): Percentage of transient masks.
        query_dims (list): Dimensions used as query in the model.
        key_dims (list): Dimensions used as key in the model.
    """
    def __init__(self, 
                 mask_extended=None, 
                 mask_transient_size=1, mask_transient_perc=None, 
                 query_dims: list = None,
                 key_dims: list = None, 
                 **kwargs):
        # Initialize with base configuration and additional parameters
        super().__init__(model_folder="RebarDist", **kwargs)
        self.mask_extended = mask_extended
        self.mask_transient_perc = mask_transient_perc
        self.mask_transient_size = mask_transient_size
        self.query_dims = query_dims
        self.key_dims = key_dims

class Model(BaseModelClass):
    """
    Model class for training and evaluating the Crossmodal RebarDist model.

    Attributes:
        optimizer (torch.optim.Optimizer): Optimizer for model training.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize optimizer with model parameters and learning rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
    
    def setup_dataloader(self, X, y, train: bool) -> torch.utils.data.DataLoader:
        """
        Setup data loader for training or evaluation.

        Args:
            X: Input data.
            y: Corresponding labels.
            train (bool): Flag indicating training mode.

        Returns:
            torch.utils.data.DataLoader: Configured data loader.
        """
        # Initialize dataset with masking parameters
        dataset = rebarcrossattn_maskdataset(
            path=X,
            data_normalizer=self.data_normalizer,
            data_clipping=self.data_clipping,
            timelen=self.config.timelen,
            mask_extended=self.config.mask_extended,
            mask_transient_size=self.config.mask_transient_size,
            mask_transient_perc=self.config.mask_transient_perc,
        )
        # Create data loader
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=train, num_workers=torch.get_num_threads()
        )
        return loader
    
    def create_state_dict(self, epoch: int, test_loss) -> dict:
        """
        Create state dictionary for checkpointing.

        Args:
            epoch (int): Current epoch number.
            test_loss: Test loss value.

        Returns:
            dict: State dictionary containing model and optimizer states.
        """
        state_dict = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "test_loss": test_loss,
            "epoch": epoch
        }
        return state_dict

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        """
        Run a single epoch of training or evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader for the current epoch.
            train (bool): Flag indicating training mode.

        Returns:
            tuple: Total loss and additional printouts (empty dictionary).
        """
        self.net.train(mode=train)  # Set model training mode
        self.optimizer.zero_grad()  # Reset gradients

        with torch.set_grad_enabled(train):
            total_loss = 0 

            # Iterate through batches
            for x_original, mask_0ismissing in tqdm(dataloader, desc="Training" if train else "Evaluating", leave=False):
                query = x_original[:, :, self.config.query_dims].to(self.device)
                key = x_original[:, :, self.config.key_dims].to(self.device)
                mask_0ismissing = mask_0ismissing[:, :, self.config.query_dims]
                
                # Perform forward pass and obtain reconstruction
                reconstruction, attn_weights = self.net(query_in=query, mask=mask_0ismissing.to(self.device), key_in=key)

                # Handle different stride cases for masking
                if self.net.stride != 1:
                    mask_0ismissing_downsamp = torch.clone(mask_0ismissing[:, ::self.net.stride])
                    mask_0ismissing_samesamp = torch.ones(mask_0ismissing.shape).bool()
                    mask_0ismissing_samesamp[:, ::self.net.stride] = mask_0ismissing[:, ::self.net.stride]
                else:
                    mask_0ismissing_downsamp, mask_0ismissing_samesamp = mask_0ismissing, mask_0ismissing

                # Compute reconstruction loss
                reconstruct_loss = torch.sum(torch.square(reconstruction[~mask_0ismissing_downsamp] - query[~mask_0ismissing_samesamp].cuda()))

                if train:
                    reconstruct_loss.backward()  # Backpropagate loss
                    self.optimizer.step()  # Update model parameters
                    self.optimizer.zero_grad()  # Reset gradients

                # Accumulate total loss
                total_loss += reconstruct_loss.item() / torch.sum(~mask_0ismissing)

            return total_loss, {}
        
    def calc_distance(self, anchor: torch.Tensor, candidate: torch.Tensor):
        """
        Calculate the distance between anchor and candidate using the model.

        Args:
            anchor (torch.Tensor): Anchor tensor.
            candidate (torch.Tensor): Candidate tensor.

        Returns:
            torch.Tensor: Computed reconstruction loss as distance.
        """
        self.net.eval()
        with torch.no_grad():
            query = anchor[:, :, self.config.query_dims].to(self.device)
            key = candidate[:, :, self.config.key_dims].to(self.device)

            # Initialize mask for missing values
            mask_0ismissing = torch.ones(query.shape, dtype=bool).cuda()
            inds = np.arange(query.shape[1])
            inds_chosen = np.random.choice(inds, query.shape[1] // 2, replace=False)
            mask_0ismissing[:, inds_chosen] = 0

            # Perform forward pass
            reconstruction, _ = self.net(query_in=query, mask=mask_0ismissing.to(self.device), key_in=key)

            # Handle different stride cases for masking
            if self.net.stride != 1:
                mask_0ismissing_downsamp = torch.clone(mask_0ismissing[:, ::self.net.stride])
                mask_0ismissing_samesamp = torch.ones(mask_0ismissing.shape).bool()
                mask_0ismissing_samesamp[:, ::self.net.stride] = mask_0ismissing[:, ::self.net.stride]
            else:
                mask_0ismissing_downsamp, mask_0ismissing_samesamp = mask_0ismissing, mask_0ismissing

            # Compute reconstruction loss
            reconstruct_loss = torch.sum(
                torch.square(reconstruction[~mask_0ismissing_downsamp].view(query.shape[0], -1, query.shape[-1]) - 
                             query[~mask_0ismissing_samesamp].view(query.shape[0], -1, query.shape[-1]).cuda()), 
                dim=(1, 2)
            )
        
        self.net.train()

        return reconstruct_loss

class rebarcrossattn_maskdataset(OnTheFly_FolderNpyDataset):
    """
    Custom dataset class for loading and masking data.

    Attributes:
        mask_extended (int): Length of the mask to be applied.
        mask_transient_size (int): Size of transient masks.
        mask_transient_perc (float): Percentage of transient masks.
    """
    def __init__(self, path, 
                 data_clipping=None,
                 data_normalizer=None,
                 mask_extended=None, 
                 mask_transient_size=1,
                 mask_transient_perc=None,
                 timelen=None):
        'Initialization'
        # Initialize base dataset class
        super().__init__(path, data_clipping=data_clipping, data_normalizer=data_normalizer, timelen=timelen)
        self.mask_extended = mask_extended
        self.mask_transient_perc = mask_transient_perc
        self.mask_transient_size = mask_transient_size
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        x_original = super().__getitem__(idx)["signal"]

        # Ensure signal has two dimensions
        if len(x_original.shape) == 1:
            x_original = x_original[:, np.newaxis]
        
        time_length = x_original.shape[0]
        mask_0ismissing = torch.ones(x_original.shape, dtype=torch.bool)

        # Apply extended mask or transient mask
        if self.mask_extended:
            start_idx = np.random.randint(time_length - self.mask_extended)
            mask_0ismissing[start_idx:start_idx + self.mask_extended, :] = False
        else:
            allinds = np.arange(0, time_length)
            num_splits = time_length // self.mask_transient_size
            grouped_allinds = np.array_split(allinds, num_splits)

            idxtomask = np.random.choice(len(grouped_allinds), int(len(grouped_allinds) * self.mask_transient_perc), replace=False)
            mask_0ismissing[np.concatenate([grouped_allinds[i] for i in idxtomask]), :] = False

        return x_original, mask_0ismissing