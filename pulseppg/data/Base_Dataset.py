import torch
import pathlib
import numpy as np
import pickle

#######################################################################
########################### Dataset Configs ###########################
#######################################################################

class Base_DatasetConfig:
    """Base configuration class for datasets.

    This class holds common settings that are shared across all dataset types,
    such as data locations and preprocessing options.

    Attributes:
        data_folder (str): The root directory where the dataset is stored.
        type (str): The type of the dataset configuration, e.g., 'supervised' or 'ssl'.
        data_normalizer_path (str): The file path to the pickled data normalizer dictionary.
            The normalizer dictionary should map user IDs to a tuple containing the
            mean, standard deviation, and clipping threshold for their signals.
        data_clipping (bool): A flag indicating whether to apply clipping to the signal
            based on the values in the normalizer.
    """
    def __init__(
        self,
        data_folder: str,
        data_normalizer_path: str = False,
        data_clipping: bool = False,
    ):
        self.data_folder = data_folder
        self.type = None

        self.data_normalizer_path = data_normalizer_path
        self.data_clipping = data_clipping


class SupervisedDataConfig(Base_DatasetConfig):
    """Configuration class for supervised learning datasets.

    This class extends `Base_DatasetConfig` with attributes required for
    supervised learning tasks, such as specifying feature and label annotations.

    Attributes:
        X_annotates (List[str]): A list of strings identifying the input features.
        y_annotate (str): A string identifying the target label.
        type (str): The type of dataset, set to "supervised".
    """
    def __init__(self, X_annotates: list = [""], y_annotate: str = "", **kwargs):
        super().__init__(**kwargs)

        self.X_annotates = X_annotates 
        self.y_annotate = y_annotate

        self.type = "supervised"


class SSLDataConfig(Base_DatasetConfig):
    """Configuration class for self-supervised learning (SSL) datasets.

    This class extends `Base_DatasetConfig` for SSL tasks. It primarily serves
    to set the configuration type to "ssl".

    Attributes:
        type (str): The type of dataset, set to "ssl".
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type = "ssl"


#######################################################################
########################### Dataset Classes ###########################
#######################################################################

class Base_Dataset(torch.utils.data.Dataset):
    """A base `torch.utils.data.Dataset` class for handling wearable sensor data.

    This class provides core functionality, including loading a data normalizer
    and applying user-specific normalization and clipping to signals. It is
    intended to be subclassed by more specific dataset implementations.

    Attributes:
        config (Base_DatasetConfig): The configuration object for the dataset.
        data_normalizer (dict): A dictionary mapping user IDs to their normalization
            parameters (mean, std, clipping_threshold).
    """
    def __init__(self, data_config: Base_DatasetConfig):
        super().__init__()
        self.data_config = data_config
        if data_config.data_normalizer_path:
            # TODO add version to construct data normalizer if file isn't there
            # this stores a dictionary where the key is the uid and the value is a 
            # 3-tuple, with user's mean, stddev, and clipping border
            with open(data_config.data_normalizer_path, 'rb') as f:
                self.data_normalizer = pickle.load(f)

    def normalize_and_clip(self, signal, user_file_path):
        """Applies user-specific Z-score normalization and optional clipping.

        The method extracts a user ID (uid) from the file path, retrieves the
        corresponding normalization stats (mean, std) and clipping threshold,
        and applies them to the input signal.

        Args:
            signal (np.ndarray): The raw input signal to be processed.
            user_file_path (str): The file path of the signal, which is used to
                extract the user ID. The UID is assumed to follow a 'train/',
                'val/', or 'test/' directory in the path.

        Returns:
            np.ndarray: The normalized and optionally clipped signal.
        """
        if self.data_config.data_normalizer_path:
            user_file_path = str(user_file_path).split('/')
            if 'train' in user_file_path:
                uid_idx = user_file_path.index('train') + 1
            elif 'val' in user_file_path:
                uid_idx = user_file_path.index('val') + 1
            elif 'test' in user_file_path:
                uid_idx = user_file_path.index('test') + 1

            uid = user_file_path[uid_idx]
            user_ppg_mean = self.data_normalizer[uid][0]
            user_ppg_std = self.data_normalizer[uid][1]
            clip_tr = self.data_normalizer[uid][2]
            
            if self.data_config.data_clipping:
                signal = np.where(signal > clip_tr, clip_tr, signal)
            signal = signal - user_ppg_mean
            signal = signal / user_ppg_std 
        
        return signal


class OnTheFly_FolderNpyDataset(Base_Dataset):
    """A dataset that loads `.npy` files from a folder structure on the fly.

    This class scans a directory for all `.npy` files and loads them one by one
    when requested by the data loader. It inherits normalization logic from
    `Base_Dataset`.

    Attributes:
        path (str): The root directory to search for `.npy` files.
        filelist (List[pathlib.Path]): A list of all `.npy` file paths found.
        length (int): The total number of files found.
    """
    def __init__(self, data_config: Base_DatasetConfig, path: list):
        super().__init__(data_config)
        self.path = path
        self.filelist = list(pathlib.Path(path).rglob("*.npy"))
        self.length = len(self.filelist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        signal = np.load(self.filelist[idx]).astype(np.float32).copy()
        filepath = self.filelist[idx]

        signal = self.normalize_and_clip(signal, filepath)

        output_dict = {"signal": signal, "filepath": str(filepath)}

        return output_dict
