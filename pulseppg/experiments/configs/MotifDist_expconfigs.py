from pulseppg.models.Base_Model import Base_ModelConfig
from pulseppg.nets.Base_Nets import Base_NetConfig

# from eval.Base_Eval import Base_EvalConfig
from pulseppg.data.Base_Dataset import SSLDataConfig, SupervisedDataConfig

from pulseppg.models.MotifDist.MotifDist_Model import MotifDist_ModelConfig, Mask_DatasetConfig


allmotifdist_expconfigs = {}


allmotifdist_expconfigs["motifdist"] = MotifDist_ModelConfig(
    query_dims = [0],
    key_dims =  [0],

    data_config=Mask_DatasetConfig(
        # data_folder="pulseppg/data/datasets/dummydataset",
        data_folder="/disk1/maxmithun/harmfulstressors/data/ppg_acc_np/",
        data_normalizer_path = "/disk1/maxmithun/harmfulstressors/data/ppg_acc_np/dict_user_ppg_mean_std_per.pickle", 
        data_clipping = True, 
        mask_extended = 300
    ),

    net_config=Base_NetConfig(
        net_folder="CrossAttn",
        net_file="CrossAttn_Net",
        params={
            "query_dimsize": 1,
            "key_dimsize": 1,
            "kernel_size": 15,
            "embed_dim": 64,
            "double_receptivefield": 5,
            "stride":10,
        },
    ),

    epochs=20,
    lr=0.001,
    batch_size=16,
    save_epochfreq=10,
) # 24_1_16_ppgdist_100day in the original file