from pulseppg.models.Base_Model import Base_ModelClass, Base_ModelConfig
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from pulseppg.utils.imports import import_model


class RelCon_ModelConfig(Base_ModelConfig):
    def __init__(
        self,
        motifdist_expconfig_key: str,
        withinuser_cands: int,
        tau=0.1,
        encoder_dims=...,
        **kwargs
    ):
        super().__init__(model_folder="RelCon", model_file="RelCon_Model", **kwargs)
        self.motifdist_expconfig_key = motifdist_expconfig_key
        self.withinuser_cands = withinuser_cands

        self.tau = tau
        self.encoder_dims = encoder_dims


class Model(Base_ModelClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

        from pulseppg.experiments.configs.MotifDist_expconfigs import (
            allmotifdist_expconfigs,
        )

        motifdist_config = allmotifdist_expconfigs[self.config.motifdist_expconfig_key]

        motifdist_config.set_rundir(self.config.motifdist_expconfig_key)
        motifdist_config.set_device(self.config.device)
        self.motifdist = import_model(model_config=motifdist_config, reload_ckpt="best")
        self.motifdist.net = self.motifdist.net.cuda()

    def setup_dataloader(self, data_config, X, y, train: bool) -> torch.utils.data.DataLoader:
        dataset = RelCon_ValidCandFolders_Dataset(
            data_config,
            X,
            withinuser_cands=self.config.withinuser_cands,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=torch.get_num_threads(),
        )

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
                anchor_signal = out_dict["signal"]
                withinuser_cand_signals = out_dict["withinuser_cand_signals"]

                (
                    bs,
                    withinuser_candsetsize,
                    length,
                    channels,
                ) = withinuser_cand_signals.shape
                distances = []
                for idx in range(1, bs):
                    # compare anchor_i with anchor_(i+idx)
                    rotated_anchor_signal = torch.cat(
                        (anchor_signal[idx:, :], anchor_signal[:idx, :]), dim=0
                    )
                    distance = self.motifdist.calc_distance(
                        anchor=anchor_signal.cuda(),
                        candidate=rotated_anchor_signal.cuda(),
                    )
                    distances.append(distance)

                for idx in range(withinuser_cand_signals.shape[1]):
                    distance = self.motifdist.calc_distance(
                        anchor=anchor_signal.cuda(),
                        candidate=withinuser_cand_signals[:, idx, :].cuda(),
                    )
                    distances.append(distance)

                distances = torch.stack(
                    distances
                )  # shape [(bs-1)+candset_sizes, batch_size]
                # sort candidate set based on distances
                _, sortedinds = torch.sort(
                    distances, dim=0
                )  # ascending order, distances increasing.

                # this should be a BS x Channel output
                emb_ancs = self.net(
                    anchor_signal[:, :, self.config.encoder_dims].transpose(1, 2).cuda()
                )
                emb_withinuser_cands = self.net(
                    withinuser_cand_signals[:, :, :, self.config.encoder_dims]
                    .view(bs * withinuser_candsetsize, length, -1)
                    .transpose(1, 2)
                    .cuda()
                )
                emb_withinuser_cands = emb_withinuser_cands.view(
                    bs, withinuser_candsetsize, -1
                )

                loss = relative_contrastive_loss(
                    emb_ancs,
                    emb_withinuser_cands,
                    sortedinds=sortedinds,
                    tau=self.config.tau,
                )

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()

            return total_loss, {}


def relative_contrastive_loss(emb_ancs, emb_withinuser_cands, sortedinds, tau=1):
    # sortedinds bels is length (BS-1)+Candsetsize x BS, bc it tells us ordering of the cand samples
    # sorted is ascending, such that most positive (smallest dist) is first
    # emb_ancs shape [BS, channels]
    # cands shape [BS, (BS-1)+candset_size, channels]
    bs, candset_size, channels = emb_withinuser_cands.shape

    loss = torch.zeros(bs, device=emb_ancs.device)
    for batch_idx in range(bs):
        emb_anc_idx = (
            emb_ancs[batch_idx, :].contiguous().view(1, -1)
        )  # shape [1, channels]

        # obtain candidate set of emb_ancs, and sort according to their distances to emb_ancs
        # following similar rotation of emb_ancs
        emb_cands_idx = torch.cat(
            (
                emb_ancs[batch_idx + 1 :, :],
                emb_ancs[:batch_idx, :],
                emb_withinuser_cands[batch_idx, :],
            ),
            dim=0,
        )
        # shape [(BS-1)+emb_withinuser_candset_size, channels]
        emb_cands_idx_revsorted = emb_cands_idx[
            torch.flip(sortedinds[:, batch_idx], dims=(0,))
        ]

        sim_revsorted = (
            F.cosine_similarity(emb_anc_idx, emb_cands_idx_revsorted, dim=1) / tau
        )
        exp_sim_revsorted = torch.exp(sim_revsorted)
        # softmax = e^(matrix - logaddexp(matrix)) = E^matrix / sumexp(matrix)
        # https://feedly.com/engineering/posts/tricks-of-the-trade-logsumexp
        cumsum_exp_sim_revsorted = torch.cumsum(exp_sim_revsorted, dim=0)

        # import pdb; pdb.set_trace()
        # check that last cumsum is equal to last noncumsum
        logsoftmax = sim_revsorted[1:] - torch.log(cumsum_exp_sim_revsorted[1:])
        loss_idx = -logsoftmax

        # import pdb; pdb.set_trace()
        # assert loss_idx[-1] == -F.log_softmax(sim_revsorted, dim=-1)[-1]

        loss[batch_idx] = torch.sum(loss_idx)

    return torch.mean(loss)


#################################################
import pathlib
from pulseppg.data.Base_Dataset import OnTheFly_FolderNpyDataset
from pulseppg.utils.datasets import filter_files_by_npy_count


class RelCon_ValidCandFolders_Dataset(OnTheFly_FolderNpyDataset):
    def __init__(self, data_config, path, withinuser_cands=5):
        "Initialization"
        super().__init__(data_config, path)
        self.filelist = filter_files_by_npy_count(self.filelist, withinuser_cands + 1)

        self.length = len(self.filelist)
        self.withinuser_cands = withinuser_cands

    def __getitem__(self, idx):
        "Generates one sample of data"
        out_dict = super().__getitem__(idx)
        filepath = out_dict["filepath"]

        parentfolder = pathlib.PosixPath(filepath).parents[0]
        signals_sameparent = set(pathlib.Path(parentfolder).rglob("*.npy"))
        signals_sameparent.remove(pathlib.Path(filepath))
        withinuser_cand_names = np.random.choice(
            list(signals_sameparent), size=self.withinuser_cands, replace=False
        )
        withinuser_cand_signals = []
        for name in withinuser_cand_names:
            withinuser_cand_signal = np.load(name).astype(np.float32)
            withinuser_cand_signals.append(withinuser_cand_signal)

        withinuser_cand_signals = np.stack(withinuser_cand_signals)
        out_dict["withinuser_cand_signals"] = withinuser_cand_signals

        return out_dict
