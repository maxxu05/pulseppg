import os
import torch
import numpy as np
import pathlib
from collections import defaultdict
import pickle


def load_data(data_config, cv_split=None):
    if cv_split is not None:
        cv_anno = f"cv{cv_split}_"
    else:
        cv_anno = ""

    if "supervised" in data_config.type:
        data_path = os.path.join(data_config.data_folder)
        final_out = []
        for mode in ["train", "val", "test"]:
            # try:
            X = []
            for X_anno in data_config.X_annotates:
                X_temp = torch.Tensor(
                    np.load(os.path.join(data_path, f"{cv_anno}{mode}_X{X_anno}.npy"))
                )
                X.append(X_temp)
            X = torch.cat(X, dim=1)
            y = np.load(
                os.path.join(data_path, f"{cv_anno}{mode}_y{data_config.y_annotate}.npy")
            )
            final_out.extend([X, y])

        return final_out

    elif data_config.type == "ssl":

        final_out = [
            os.path.join(data_config.data_folder, "train"),
            None,
            os.path.join(data_config.data_folder, "val"),
            None,
            os.path.join(data_config.data_folder, "test"),
            None,
        ]

        return final_out


def filter_files_by_npy_count(files, min_py_files=5):
    """
    Filters a list of files, keeping only those whose parent directory contains at least a certain number of .npy files.

    :param files: List of file paths (absolute paths).
    :param min_py_files: Minimum number of .npy files required in the parent directory.
    :return: List of file paths that meet the criteria.
    """
    # Count .npy files in each parent directory
    directory_npy_count = defaultdict(int)
    for file in files:
        file_path = pathlib.Path(file)
        if file_path.suffix == ".npy":
            directory_npy_count[file_path.parent] += 1

    # Filter files based on the precomputed .npy counts
    qualifying_files = [
        file
        for file in files
        if directory_npy_count[pathlib.Path(file).parent] >= min_py_files
    ]

    return qualifying_files
