import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
import os

from pulseppg.data.Base_Dataset import Base_DatasetConfig
from pulseppg.nets.Base_Nets import Base_NetConfig

# from eval.Base_Eval import Base_EvalConfig


class Base_ModelConfig:
    def __init__(
        self,
        # model parameters
        model_folder: str,
        model_file: str,
        # data parameters
        data_config: Base_DatasetConfig,
        # network configuration
        net_config: Base_NetConfig = None,
        # model training parameters
        epochs=50,
        lr=0.001,
        batch_size=16,
        save_epochfreq=100,
        # experiment params
        seed=1234,
        num_threads=-1,
        ####################################
        # eval stuff
        eval_configs=[],
        #####################################
    ):
        self.model_folder = model_folder
        self.model_file = model_file

        self.data_config = data_config

        self.net_config = net_config

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.save_epochfreq = save_epochfreq
        self.seed = seed
        self.num_threads = num_threads

        self.eval_configs = eval_configs

        self.device = None
        self.input_dims = None

    def set_device(self, device):
        self.device = device

    def set_inputdims(self, dims):
        self.input_dims = dims

    def set_rundir(self, run_dir):
        self.run_dir = run_dir


class Base_ModelClass:
    def __init__(
        self,
        config: Base_ModelConfig,
        train_data=None,
        train_labels=None,
        val_data=None,
        val_labels=None,
        test_data=None,
        test_labels=None,
        seed=10,
        resume_on=False,
    ):
        from pulseppg.utils.utils import set_seed

        set_seed(seed)

        # import pdb; pdb.set_trace()
        self.config = config

        self.train_data, self.train_labels = train_data, train_labels
        self.val_data, self.val_labels = val_data, val_labels
        self.test_data, self.test_labels = test_data, test_labels

        self.model_file = config.model_file

        self.run_dir = os.path.join("pulseppg/experiments/out", config.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.device = config.device
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.save_epochfreq = config.save_epochfreq

        self.resume_on = resume_on

        from pulseppg.utils.imports import import_net

        if config.net_config is not None:
            self.net = import_net(config.net_config).to(self.device)

    @abstractmethod
    def setup_dataloader(
        self, data_config, data: torch.Tensor, labels: torch.Tensor, train: bool
    ) -> torch.utils.data.DataLoader:
        ...

    @abstractmethod
    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        ...

    @abstractmethod
    def create_state_dict(self, epoch: int):
        ...

    def fit(self):
        from pulseppg.utils.utils import printlog

        printlog(f"Begin Training {self.model_file}", self.run_dir)

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        train_loader = self.setup_dataloader(
            data_config = self.config.data_config, X=self.train_data, y=self.train_labels, train=True
        )
        val_loader = self.setup_dataloader(
            data_config = self.config.data_config, X=self.val_data, y=self.val_labels, train=False
        )

        train_loss_list, val_loss_list = [], []
        best_val_loss = np.inf

        start_epoch = 0
        if (self.resume_on) and (
            os.path.exists(os.path.join(self.run_dir, "checkpoint_latest.pkl"))
        ):
            state_dict = self.load(ckpt="latest", return_state_dict=True)
            self.optimizer.load_state_dict(state_dict["optimizer"])
            start_epoch = state_dict["epoch"] + 1
            printlog(
                f"Resuming model from epoch {state_dict['epoch']}, with {self.epochs-start_epoch} additional epochs remaining for training",
                self.run_dir,
            )
            if os.path.exists(os.path.join(self.run_dir, "checkpoint_best.pkl")):
                best_state_dict = torch.load(
                    f"{self.run_dir}/checkpoint_best.pkl", map_location=self.device
                )
                best_val_loss = best_state_dict["test_loss"]
            else:
                best_val_loss = state_dict["test_loss"]

        for epoch in tqdm(range(start_epoch, self.epochs), desc=f"{self.run_dir} fit:"):
            train_loss, train_printouts = self.run_one_epoch(train_loader, train=True)
            train_loss_list.append(train_loss)

            val_loss, val_printouts = self.run_one_epoch(val_loader, train=False)
            val_loss_list.append(val_loss)

            state_dict = self.create_state_dict(epoch, val_loss)
            if epoch % self.save_epochfreq == 0:
                torch.save(state_dict, f"{self.run_dir}/checkpoint_epoch{epoch}.pkl")
            if epoch == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state_dict, f"{self.run_dir}/checkpoint_best.pkl")
            torch.save(state_dict, f"{self.run_dir}/checkpoint_latest.pkl")

            printoutstring = (
                f"Epoch #{epoch}: Loss/Train={train_loss:5f} | Loss/Val={val_loss:5f}"
            )
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)

            # for key in train_printouts.keys():
            #     writer.add_scalar(f'{key}/Train', train_printouts[key], epoch)
            #     writer.add_scalar(f'{key}/Val', val_printouts[key], epoch)
            #     printoutstring += f'\n {key}/Train: {train_printouts[key]} | {key}/Val: {val_printouts[key]} || '
            printlog(printoutstring, self.run_dir)

    def test(self):
        from pulseppg.utils.utils import printlog

        printlog(f"Loading Best From Training", self.run_dir)
        self.load()

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        test_loader = self.setup_dataloader(
            X=self.test_data, y=self.test_labels, train=False
        )

        test_loss_list = []
        test_loss, test_printouts = self.run_one_epoch(test_loader, train=False)
        test_loss_list.append(test_loss)

        epoch = 0
        printoutstring = f"Loss/Test={test_loss:5f}"
        writer.add_scalar("Loss/Test", test_loss, epoch)

        for key in test_printouts.keys():
            writer.add_scalar(f"{key}/Test", test_printouts[key], epoch)
            printoutstring += f"\n {key}/Test: {test_printouts[key]} || "
        printlog(printoutstring, self.run_dir)

        return test_printouts

    def load(self, ckpt="best", return_state_dict=False):
        from pulseppg.utils.utils import printlog

        state_dict = torch.load(f'{self.run_dir}/checkpoint_{ckpt}.pkl', map_location=self.device)

        print(self.net.load_state_dict(state_dict["net"]))
        printlog(f"Reloading {self.model_file} Model's ckpt {ckpt}, which is from epoch {state_dict['epoch']}", self.run_dir)
        if return_state_dict:
            return state_dict
