from pulseppg.nets.Base_Nets import Base_NetConfig

from pulseppg.eval.Base_Eval import Base_EvalConfig
from pulseppg.data.Base_Dataset import SSLDataConfig, SupervisedDataConfig

from pulseppg.models.RelCon.RelCon_Model import RelCon_ModelConfig


allrelcon_expconfigs = {}


# allrelcon_expconfigs["25_3_8_relcon"] = RelCon_ModelConfig(
#     withinuser_cands=20,
#     motifdist_expconfig_key="25_3_8_motifdist",
#     data_config=SSLDataConfig(
#         data_folder="pulseppg/data/datasets/dummydataset",
#     ),
#     net_config=Base_NetConfig(
#         net_folder="ResNet1D",
#         net_file="ResNet1D_Net",
#         params={
#             "init_conv": {
#                 "in_channels": 3,
#                 "base_filters": 64,
#                 "kernel_size": 7,
#                 "stride": 2,
#                 "padding": 0,
#             },
#             "init_maxpool": {
#                 "kernel_size": 3,
#                 "stride": 2,
#             },
#             "blocks": [
#                 {
#                     "n": 3,
#                     "in_channels": 64,
#                     "base_filters": 64,
#                     "kernel_size": 3,
#                     "stride": 1,
#                 },
#                 {
#                     "n": 4,
#                     "in_channels": 64,
#                     "base_filters": 128,
#                     "kernel_size": 4,
#                     "stride": 1,
#                 },
#                 {
#                     "n": 6,
#                     "in_channels": 128,
#                     "base_filters": 256,
#                     "kernel_size": 4,
#                     "stride": 1,
#                 },
#             ],
#             "finalpool": "avg",
#         },
#     ),
#     epochs=100,
#     lr=0.000001,
#     batch_size=64,
#     save_epochfreq=10,
#     eval_configs = [
#             Base_EvalConfig(
#                 name="HHAR | Linear Probe | Comparison against SSL Benchmark", 
#                 model_folder="Classify",
#                 model_file="linear_probe",
#                 cv_splits = 5,
#                 # data parameters
#                 data_config=SupervisedDataConfig(
#                     data_folder="pulseppg/data/datasets/sslbench/hhar/processed",
#                 ),
#             ),
#             Base_EvalConfig(
#                 name="Motionsense | Linear Probe | Comparison against SSL Benchmark", 
#                 model_folder="Classify",
#                 model_file="linear_probe",
#                 cv_splits = 5,
#                 # data parameters
#                 data_config=SupervisedDataConfig(
#                     data_folder="pulseppg/data/datasets/sslbench/motionsense/processed",
#                 ),
#             ),
#             Base_EvalConfig(
#                 name="PAMAP2 | Linear Probe | Comparison against SSL Benchmark", 
#                 model_folder="Classify",
#                 model_file="linear_probe",
#                 cv_splits = 5,
#                 # data parameters
#                 data_config=SupervisedDataConfig(
#                     data_folder="pulseppg/data/datasets/sslbench/pamap2/processed",
#                 ),
#             ),
#             Base_EvalConfig(
#                 name="PAMAP2 | MLP Probe | Comparison against Prior Pre-trained Model", 
#                 model_folder="Classify",
#                 model_file="MLP_probe",
#                 cv_splits = 8,
#                 # data parameters
#                 data_config=SupervisedDataConfig(
#                     data_folder="pulseppg/data/datasets/priorpt/pamap2/processed",
#                 ),
#             ),
#             Base_EvalConfig(
#                 name="PAMAP2 | MLP Fine-Tune | Comparison against Prior Pre-trained Model", 
#                 model_folder="Classify",
#                 model_file="MLP_finetune",
#                 cv_splits = 8,
#                 # data parameters
#                 data_config=SupervisedDataConfig(
#                     data_folder="pulseppg/data/datasets/priorpt/pamap2/processed",
#                 ),
#                 evalnetparams = {'embed_dim': 256,
#                                  "mlp_dim": 512,
#                                  "class_num": 8},
#                 epochs=10, lr=.001, batch_size=16, save_epochfreq=5,
#             ),
#             Base_EvalConfig(
#                 name="Opportunity | MLP Probe | Comparison against Prior Pre-trained Model", 
#                 model_folder="Classify",
#                 model_file="MLP_probe",
#                 cv_splits = 4,
#                 # data parameters
#                 data_config=SupervisedDataConfig(
#                     data_folder="pulseppg/data/datasets/priorpt/opportunity/processed",
#                 ),
#             ),
#             Base_EvalConfig(
#                 name="Opportunity | MLP Fine-Tune | Comparison against Prior Pre-trained Model", 
#                 model_folder="Classify",
#                 model_file="MLP_finetune",
#                 cv_splits = 4,
#                 # data parameters
#                 data_config=SupervisedDataConfig(
#                     data_folder="pulseppg/data/datasets/priorpt/opportunity/processed",
#                 ),
#                 evalnetparams = {'embed_dim': 256,
#                                  "mlp_dim": 512,
#                                  "class_num": 8},
#                 epochs=10, lr=.001, batch_size=16, save_epochfreq=5,
#             ),
#     ]
# )
