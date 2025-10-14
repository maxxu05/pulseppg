import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
import os
from pulseppg.models.Base_Model import Base_ModelConfig
from pulseppg.models.Base_Model import Base_ModelClass

class Base_EvalConfig(Base_ModelConfig):
    def __init__(self, 
                 # model parameters
                 model_folder: str,
                 model_file: str,
                 # data parameters
                 data_config, 
                 name: str = "",
                 # model training parameters
                 epochs=50, lr=0.001, batch_size=16, save_epochfreq=100,
                 ##############################
                 pretrain_epoch="best",
                 evalnetparams = {},
                 num_threads=-1, verbose=1,
                 ):
        self.name = name

        self.model_folder = model_folder
        self.model_file = model_file

        self.data_config = data_config 

        self.epochs = epochs
        self.lr = lr 
        self.batch_size = batch_size
        self.save_epochfreq = save_epochfreq

        self.device = None
        self.input_dims = None

        self.net_config = None
        self.evalnetparams = evalnetparams
        self.pretrain_epoch = pretrain_epoch

        self.num_threads=num_threads
        self.verbose=verbose
    
class Base_EvalClass(Base_ModelClass):
    def __init__(
        self,
        *args,
        **kwargs
    ): 
        super().__init__(*args,**kwargs)
        

    def create_state_dict(self, epoch: int, test_loss) -> dict:
        state_dict = {"net": self.net.state_dict(),
                      "trained_net": self.trained_net.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "test_loss": test_loss,
                      "epoch": epoch}

        return state_dict

    def setup_eval(self, trained_net):
        self.trained_net = trained_net

    
    def load(self, ckpt="best", return_state_dict = False):
        
        from utils.utils import printlog

        state_dict = torch.load(f'{self.run_dir}/checkpoint_{ckpt}.pkl', map_location=self.device)

        print(self.net.load_state_dict(state_dict["net"]))
        print(self.trained_net.load_state_dict(state_dict["trained_net"]))

        printlog(f"Reloading {self.model_file} Model's ckpt {ckpt}, which is from epoch {state_dict['epoch']}", self.run_dir)


        if return_state_dict:
            return state_dict
