import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_ecg._preprocessors import Normalize
from utils import resample_batch_signal, preprocess_one_ppg_signal

class SDBDataProcessor:
    def __init__(self, download_dir: str, split_dir: str, fs_target: int):
        self.download_dir = download_dir
        self.split_dir = split_dir
        self.fs = 62.5
        self.fs_target = fs_target

        self.filenames = sorted([f for f in os.listdir(self.download_dir) if f.startswith("subject")])
        self.train_subjects = np.array(pd.read_csv(os.path.join(split_dir, "train.csv"))["subjectNumber"])
        self.val_subjects = np.array(pd.read_csv(os.path.join(split_dir, "val.csv"))["subjectNumber"])
        self.test_subjects = np.array(pd.read_csv(os.path.join(split_dir, "test.csv"))["subjectNumber"])
        self.labels = pd.read_csv(os.path.join(download_dir, "AHI.csv"))

        self.norm = Normalize(method="z-score")

    def process_data(self):
        chunk_size = int(self.fs_target * 10)

        train_X, train_y = [], []
        val_X, val_y = [], []
        test_X, test_y = [], []

        for fname in tqdm(self.filenames):
            df = pd.read_csv(os.path.join(self.download_dir, fname))
            signal = np.array(df["pleth"])

            signal, _ = self.norm.apply(signal, fs=self.fs)
            signal, _, _, _ = preprocess_one_ppg_signal(waveform=signal, frequency=self.fs)

            sig_r = resample_batch_signal(signal, fs_original=self.fs, fs_target=self.fs_target, axis=0)
            num_chunks = len(sig_r) // chunk_size
            chunks = np.array_split(sig_r[:num_chunks * chunk_size], num_chunks) if num_chunks > 0 else []

            m = re.search(r"subject(\d+)", fname)
            if not m:
                continue
            subject_id = int(m.group(1))

            try:
                label = self.labels.iloc[np.where(self.labels["subjectNumber"] == subject_id)[0], 1].item()
            except (IndexError, ValueError):
                continue

            if subject_id in self.train_subjects:
                train_X.extend(chunks); train_y.extend([label] * len(chunks))
            elif subject_id in self.val_subjects:
                val_X.extend(chunks); val_y.extend([label] * len(chunks))
            elif subject_id in self.test_subjects:
                test_X.extend(chunks); test_y.extend([label] * len(chunks))
            else:
                continue

        train_X = np.array(train_X)[:, None, :] if len(train_X) > 0 else np.zeros((0,1,chunk_size))
        val_X = np.array(val_X)[:, None, :] if len(val_X) > 0 else np.zeros((0,1,chunk_size))
        test_X = np.array(test_X)[:, None, :] if len(test_X) > 0 else np.zeros((0,1,chunk_size))

        train_y = np.array(train_y)
        val_y = np.array(val_y)
        test_y = np.array(test_y)

        # Save files named with current target frequency
        np.save(os.path.join(self.download_dir, f"train_X_ppg_{self.fs_target}Hz.npy"), train_X)
        np.save(os.path.join(self.download_dir, f"train_y_sdb.npy"), train_y)

        np.save(os.path.join(self.download_dir, f"val_X_ppg_{self.fs_target}Hz.npy"), val_X)
        np.save(os.path.join(self.download_dir, f"val_y_sdb.npy"), val_y)

        np.save(os.path.join(self.download_dir, f"test_X_ppg_{self.fs_target}Hz.npy"), test_X)
        np.save(os.path.join(self.download_dir, f"test_y_sdb.npy"), test_y)


def main(newhz: int, download_dir: str, split_dir: str):
    processor = SDBDataProcessor(download_dir=download_dir, split_dir=split_dir, fs_target=newhz)
    processor.process_data()


if __name__ == "__main__":
    download_dir = "../../data/sdb"
    split_dir = "../../data/papageisplits/PaPaGei/SDB"
    main(newhz=50, download_dir=download_dir, split_dir=split_dir)
    main(newhz=125, download_dir=download_dir, split_dir=split_dir)