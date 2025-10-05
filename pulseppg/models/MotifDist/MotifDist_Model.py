from pulseppg.models.Base_Model import Base_ModelConfig, Base_ModelClass
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class MotifDist_ModelConfig(Base_ModelConfig):
    def __init__(
        self, query_dims: list = [0, 1, 2], key_dims: list = [0, 1, 2], **kwargs
    ):
        super().__init__(
            model_folder="MotifDist", model_file="MotifDist_Model", **kwargs
        )
        self.query_dims = query_dims
        self.key_dims = key_dims


class Model(Base_ModelClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

    def setup_dataloader(self, X, y, train: bool) -> torch.utils.data.DataLoader:
        dataset = crossattn_augdataset(path=X)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=train, num_workers=0
        )  # torch.get_num_threads())

        return loader

    def create_state_dict(self, epoch: int, test_loss) -> dict:
        state_dict = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "test_loss": test_loss,
            "epoch": epoch,
        }

        return state_dict

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        self.net.train(mode=train)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            total_loss = 0

            for out_dict in tqdm(
                dataloader, desc="Training" if train else "Evaluating", leave=False
            ):
                x_original = out_dict["signal"]
                x_aug = out_dict["aug_signal"]
                query = x_original[:, :, self.config.query_dims].to(self.device)
                key = x_aug[:, :, self.config.key_dims].to(self.device)

                reconstruction, attn_weights = self.net(query_in=query, key_in=key)

                reconstruct_loss = torch.sum(
                    torch.square(reconstruction - query.cuda())
                )

                if train:
                    reconstruct_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += reconstruct_loss.item()

            return total_loss, {}

    def calc_distance(self, anchor: torch.Tensor, candidate: torch.Tensor):
        self.net.eval()
        with torch.no_grad():
            query = anchor[:, :, self.config.query_dims].to(self.device)
            key = candidate[:, :, self.config.key_dims].to(self.device)

            reconstruction, attn_weights = self.net(query_in=query, key_in=key)
            reconstruct_loss = torch.sum(
                torch.square(reconstruction - query.cuda()), dim=(1, 2)
            )

        self.net.train()

        return reconstruct_loss


from pulseppg.data.Base_Dataset import OnTheFly_FolderNpyDataset
from pulseppg.models.MotifDist.utils.augmentations import (
    noise_transform,
    scaling_transform,
    rotation_transform,
    negate_transform,
    time_flip_transform,
    channel_shuffle_transform,
    time_segment_permutation_transform,
    time_warp_transform,
)


class crossattn_augdataset(OnTheFly_FolderNpyDataset):
    def __init__(self, path):
        "Initialization"
        super().__init__(path)
        self.transform_funcs = [
            noise_transform,
            scaling_transform,
            rotation_transform,
            negate_transform,
            time_flip_transform,
            channel_shuffle_transform,
            time_segment_permutation_transform,
            time_warp_transform,
        ]

    def __getitem__(self, idx):
        "Generates one sample of data"
        out_dict = super().__getitem__(idx)
        x_original = out_dict["signal"]
        time_length, channels = x_original.shape

        # 8 total transforms, following https://arxiv.org/abs/2011.11542, randomly choose 2 to apply
        transform_idx = np.random.choice(np.arange(8), 2, replace=False)

        x_transform = x_original[
            None, :
        ]  # adding fake batch dimension for transform funcs
        for i in transform_idx:
            transform_func = self.transform_funcs[i]
            x_transform = transform_func(x_transform)
        x_transform = x_transform[0, :]  # remove fake batch dimension

        out_dict["aug_signal"] = torch.Tensor(x_transform.copy())
        return out_dict
