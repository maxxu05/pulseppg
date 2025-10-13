import os
import torch
import numpy as np

from pulseppg.models.Base_Model import Base_ModelConfig


def import_net(net_config: Base_ModelConfig):
    net_folder = net_config.net_folder
    net_file = net_config.net_file
    net_module = __import__(f"pulseppg.nets.{net_folder}.{net_file}", fromlist=[""])
    net_module_class = getattr(net_module, "Net")
    net = net_module_class(**net_config.params)

    return net


def import_model(
    model_config: Base_ModelConfig,
    train_data=None,
    train_labels=None,
    val_data=None,
    val_labels=None,
    test_data=None,
    test_labels=None,
    reload_ckpt=False,
    evalmodel=False,
    resume_on=False,
):

    model_folder = model_config.model_folder
    model_file = model_config.model_file

    if not evalmodel:
        parentfolder = "models"
    else:
        parentfolder = "eval"

    model_module = __import__(
        f"pulseppg.{parentfolder}.{model_folder}.{model_file}", fromlist=[""]
    )
    model_module_class = getattr(model_module, "Model")

    model = model_module_class(
        model_config,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
        resume_on=resume_on,
    )
    if reload_ckpt:
        try:
            model.load(reload_ckpt)
        except FileNotFoundError as e:
            print(f"{reload_ckpt} not found, cannot reload checkpoint")
            print(e)

    return model
