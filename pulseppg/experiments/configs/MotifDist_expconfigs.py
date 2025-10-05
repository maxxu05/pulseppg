from pulseppg.models.Base_Model import Base_ModelConfig
from pulseppg.nets.Base_Nets import Base_NetConfig

# from eval.Base_Eval import Base_EvalConfig
from pulseppg.data.Base_Dataset import SSLDataConfig, SupervisedDataConfig

from pulseppg.models.MotifDist.MotifDist_Model import MotifDist_ModelConfig


allmotifdist_expconfigs = {}


allmotifdist_expconfigs["25_3_8_motifdist"] = MotifDist_ModelConfig(
    data_config=SSLDataConfig(
        data_folder="relcon/data/datasets/dummydataset",
    ),
    net_config=Base_NetConfig(
        net_folder="CrossAttn",
        net_file="CrossAttn_Net",
        params={
            "query_dimsize": 3,
            "key_dimsize": 3,
            "kernel_size": 15,
            "embed_dim": 64,
            "double_receptivefield": 2,
        },
    ),
    epochs=10,
    lr=0.001,
    batch_size=64,
    save_epochfreq=10,
)
