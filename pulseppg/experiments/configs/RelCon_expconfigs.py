from pulseppg.nets.Base_Nets import Base_NetConfig

from pulseppg.eval.Base_Eval import Base_EvalConfig
from pulseppg.data.Base_Dataset import SSLDataConfig, SupervisedDataConfig

from pulseppg.models.RelCon.RelCon_Model import RelCon_ModelConfig


allrelcon_expconfigs = {}

allrelcon_expconfigs["pulseppg"] = RelCon_ModelConfig(
    withinuser_cands=1,
    encoder_dims=[0],

    motifdist_expconfig_key="motifdist",

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
    eval_configs = [
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
            Base_EvalConfig(
                name="PPG-BP | Systolic BP | Linear Probe",
                model_folder="Regress",
                model_file="linear_probe",
                data_config=SupervisedDataConfig(
                   data_folder="pulseppg/data/ppgbp/",
                   X_annotates=["ppg"],
                   y_annotate="sysbp"
                ),
            ),
            # Base_EvalConfig(
            #     name="PPG-BP_diasbp",
            #     model_folder="Regress",
            #     model_file="ridgepapageireg",
            #     final_pool = None,
            #     evalnetparams = {'embed_dim': 512,
            #                      "loss": "mse"},
            #     data_config=SupervisedDataConfig(
            #        data_folder="ppgbp/",
            #        X_annotates=["ppg"],
            #        y_annotate="diasbp"
            #     ),
            # ),
            # Base_EvalConfig(
            #     name="PPG-BP_hr",
            #     model_folder="Regress",
            #     model_file="ridgepapageireg",
            #     final_pool = None,
            #     data_config=SupervisedDataConfig(
            #        data_folder="ppgbp/",
            #        X_annotates=["ppg"],
            #        y_annotate="hr"
            #     ),
            # ),
            # Base_EvalConfig(
            #     name="PPG-BP_ht_binary",
            #     model_folder="Classifier",
            #     model_file="logisticsk",
            #     final_pool = None,
            #     # data parameters
            #     data_config=SupervisedDataConfig(
            #        data_folder="ppgbp/",
            #        X_annotates=["ppg"],
            #        y_annotate="htbinary"
            #     ),
            #     # epochs=10, lr=.01, batch_size=16, save_epochfreq=5,
            # ),
    ]
) # original config called 25_1_17_relcon_ppgdist100days_c1tp1f128k11s2b12bs64lrp0001_epoch20_100daydata
