from pulseppg.models.Base_Model import Base_ModelConfig, Base_ModelClass
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class MotifDist_ModelConfig(Base_ModelConfig):
    def __init__(
        self, query_dims: list = [0], key_dims: list = [0], mask_extended: int = 300, **kwargs
    ):
        super().__init__(
            model_folder="MotifDist", model_file="MotifDist_Model", **kwargs
        )
        self.query_dims = query_dims
        self.key_dims = key_dims
        self.mask_extended = mask_extended


class Model(Base_ModelClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

    def setup_dataloader(self, data_config, X, y, train: bool) -> torch.utils.data.DataLoader:
        dataset = crossattn_maskdataset(data_config, path=X)
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
                mask_0ismissing = out_dict["mask_0ismissing"]

                query = x_original[:, :, self.config.query_dims].to(self.device)
                key = x_original[:, :, self.config.key_dims].to(self.device)

                mask_0ismissing = mask_0ismissing[:,:,self.config.query_dims].to(self.device)
                
                if self.net.stride != 1:
                    mask_0ismissing_downsamp = torch.clone(mask_0ismissing[:, ::self.net.stride])
                    mask_0ismissing_samesamp = torch.ones_like(mask_0ismissing).bool()
                    mask_0ismissing_samesamp[:, ::self.net.stride] = mask_0ismissing[:, ::self.net.stride]
                else:
                    mask_0ismissing_downsamp, mask_0ismissing_samesamp = mask_0ismissing, mask_0ismissing
                    
                reconstruction, attn_weights = self.net(query_in=query, key_in=key,
                                                        mask=mask_0ismissing)

                reconstruct_loss = torch.sum(
                    torch.square(reconstruction[~mask_0ismissing_downsamp] - 
                                 query[~mask_0ismissing_samesamp])
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


######################################################################
################## Masking Dataset Config ##################
######################################################################
from pulseppg.data.Base_Dataset import SSLDataConfig
class Mask_DatasetConfig(SSLDataConfig):
    def __init__(self, mask_extended, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_extended = mask_extended

######################################################################
################## Masking Dataset ##################
######################################################################
from pulseppg.data.Base_Dataset import OnTheFly_FolderNpyDataset
class crossattn_maskdataset(OnTheFly_FolderNpyDataset):
    def __init__(self, data_config: Mask_DatasetConfig, path):
        super().__init__(data_config, path)
        
    def __getitem__(self, idx):
        out_dict = super().__getitem__(idx)
        x_original = out_dict["signal"]
        time_length, channels = x_original.shape

        mask_0ismissing = torch.ones(x_original.shape, dtype=torch.bool)
        if self.data_config.mask_extended:
            start_idx = np.random.randint(time_length-self.data_config.mask_extended)
            mask_0ismissing[start_idx:start_idx+self.data_config.mask_extended, :] = False
        out_dict["mask_0ismissing"] = torch.Tensor(mask_0ismissing)
        return out_dict


######################################################################
################## Unused Augmentation Modification ##################
######################################################################
# from pulseppg.models.MotifDist.utils.augmentations import (
#     noise_transform,
#     scaling_transform,
#     rotation_transform,
#     negate_transform,
#     time_flip_transform,
#     channel_shuffle_transform,
#     time_segment_permutation_transform,
#     time_warp_transform,
# )
# class crossattn_augdataset(OnTheFly_FolderNpyDataset):
#     def __init__(self, data_config, path):
#         super().__init__(data_config, path)
#         self.transform_funcs = [
#             noise_transform,
#             scaling_transform,
#             rotation_transform,
#             negate_transform,
#             time_flip_transform,
#             channel_shuffle_transform,
#             time_segment_permutation_transform,
#             time_warp_transform,
#         ]

#     def __getitem__(self, idx):
#         out_dict = super().__getitem__(idx)
#         x_original = out_dict["signal"]
#         time_length, channels = x_original.shape

#         # 8 total transforms, following https://arxiv.org/abs/2011.11542, randomly choose 2 to apply
#         transform_idx = np.random.choice(np.arange(8), 2, replace=False)

#         x_transform = x_original[
#             None, :
#         ]  # adding fake batch dimension for transform funcs
#         for i in transform_idx:
#             transform_func = self.transform_funcs[i]
#             x_transform = transform_func(x_transform)
#         x_transform = x_transform[0, :]  # remove fake batch dimension

#         out_dict["aug_signal"] = torch.Tensor(x_transform.copy())
#         return out_dict
