from pulseppg.nets.Base_Nets import Base_NetConfig

from pulseppg.eval.Base_Eval import Base_EvalConfig
from pulseppg.data.Base_Dataset import SSLDataConfig, SupervisedDataConfig

from pulseppg.models.RelCon.RelCon_Model import RelCon_ModelConfig


allrelcon_expconfigs = {}

# original config called 25_3_8_relcon
allrelcon_expconfigs["25_3_8_relcon"] = RelCon_ModelConfig(
    withinuser_cands=1,
    encoder_dims=[0],

    motifdist_expconfig_key="25_10_4_motifdist",

    data_config=SSLDataConfig(
        data_folder="/disk1/maxmithun/harmfulstressors/data/ppg_acc_np/",
        data_normalizer_path = "/disk1/maxmithun/harmfulstressors/data/ppg_acc_np/dict_user_ppg_mean_std_per.pickle", 
        data_clipping = True, 
    ),

    net_config=Base_NetConfig(
        net_folder="ResNet1D",
        net_file="ResNet1D_Net",
        params = {"in_channels":1,
                  "base_filters": 128,
                  "kernel_size": 11, # 15 -> 30 -> 60 -> 120 -> 240 -> 480
                  "stride":2,
                  "groups": 1,
                  "n_block": 12,
                  "finalpool": "max"}
    ),
    epochs = 20, lr=0.0001, batch_size=16, save_epochfreq=1,
    # eval_configs = [
    #         Base_EvalConfig(
    #             name="HHAR | Linear Probe | Comparison against SSL Benchmark", 
    #             model_folder="Classify",
    #             model_file="linear_probe",
    #             cv_splits = 5,
    #             # data parameters
    #             data_config=SupervisedDataConfig(
    #                 data_folder="pulseppg/data/datasets/sslbench/hhar/processed",
    #             ),
    #         ),
    #         Base_EvalConfig(
    #             name="Motionsense | Linear Probe | Comparison against SSL Benchmark", 
    #             model_folder="Classify",
    #             model_file="linear_probe",
    #             cv_splits = 5,
    #             # data parameters
    #             data_config=SupervisedDataConfig(
    #                 data_folder="pulseppg/data/datasets/sslbench/motionsense/processed",
    #             ),
    #         ),
    #         Base_EvalConfig(
    #             name="PAMAP2 | Linear Probe | Comparison against SSL Benchmark", 
    #             model_folder="Classify",
    #             model_file="linear_probe",
    #             cv_splits = 5,
    #             # data parameters
    #             data_config=SupervisedDataConfig(
    #                 data_folder="pulseppg/data/datasets/sslbench/pamap2/processed",
    #             ),
    #         ),
    #         Base_EvalConfig(
    #             name="PAMAP2 | MLP Probe | Comparison against Prior Pre-trained Model", 
    #             model_folder="Classify",
    #             model_file="MLP_probe",
    #             cv_splits = 8,
    #             # data parameters
    #             data_config=SupervisedDataConfig(
    #                 data_folder="pulseppg/data/datasets/priorpt/pamap2/processed",
    #             ),
    #         ),
    #         Base_EvalConfig(
    #             name="PAMAP2 | MLP Fine-Tune | Comparison against Prior Pre-trained Model", 
    #             model_folder="Classify",
    #             model_file="MLP_finetune",
    #             cv_splits = 8,
    #             # data parameters
    #             data_config=SupervisedDataConfig(
    #                 data_folder="pulseppg/data/datasets/priorpt/pamap2/processed",
    #             ),
    #             evalnetparams = {'embed_dim': 256,
    #                              "mlp_dim": 512,
    #                              "class_num": 8},
    #             epochs=10, lr=.001, batch_size=16, save_epochfreq=5,
    #         ),
    #         Base_EvalConfig(
    #             name="Opportunity | MLP Probe | Comparison against Prior Pre-trained Model", 
    #             model_folder="Classify",
    #             model_file="MLP_probe",
    #             cv_splits = 4,
    #             # data parameters
    #             data_config=SupervisedDataConfig(
    #                 data_folder="pulseppg/data/datasets/priorpt/opportunity/processed",
    #             ),
    #         ),
    #         Base_EvalConfig(
    #             name="Opportunity | MLP Fine-Tune | Comparison against Prior Pre-trained Model", 
    #             model_folder="Classify",
    #             model_file="MLP_finetune",
    #             cv_splits = 4,
    #             # data parameters
    #             data_config=SupervisedDataConfig(
    #                 data_folder="pulseppg/data/datasets/priorpt/opportunity/processed",
    #             ),
    #             evalnetparams = {'embed_dim': 256,
    #                              "mlp_dim": 512,
    #                              "class_num": 8},
    #             epochs=10, lr=.001, batch_size=16, save_epochfreq=5,
    #         ),
    # ]
)
