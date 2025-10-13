import os
import re
import numpy as np
import pandas as pd
import zipfile
import requests
from tqdm import tqdm
from torch_ecg._preprocessors import Normalize
from utils import resample_batch_signal, preprocess_one_ppg_signal

class SDBDataProcessor:
    def __init__(self, zippath: str, ppgpath: str, fs_target: int):
        self.zippath = zippath
        self.ppgpath = ppgpath

        self.fs = 62.5
        self.fs_target = fs_target

        ## this was added in order to match papagei's original splits
        self.train_subjects = [1, 2, 4, 7, 9, 11, 13, 18, 19, 20, 26, 27, 30, 31, 32, 33, 35, 40, 41, 43, 44, 46, 47, 50, 51, 53, 55, 58, 59, 61, 62, 63, 64, 65, 67, 69, 70, 71, 72, 73, 76, 78, 79, 82, 86, 87, 90, 91, 92, 95, 98, 104, 105, 109, 110, 112, 114, 117, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 145, 147, 150, 153, 156, 157, 160]
        self.val_subjects = [5, 8, 10, 14, 15, 16, 22, 28, 38, 54, 57, 66, 68, 74, 80, 81, 85, 88, 89, 94, 96, 102, 103, 118, 125, 143, 146, 148, 154]
        self.test_subjects = [3, 12, 23, 24, 25, 36, 37, 42, 45, 48, 49, 52, 56, 60, 77, 83, 84, 97, 99, 101, 106, 111, 113, 115, 116, 144, 149, 151, 155, 158]

        self.norm = Normalize(method="z-score")


    def downloadextract_PPGfiles(self, redownload: bool = False) -> None:
        """
        Downloads and extracts PPG files if they do not already exist or if redownload is requested.

        :param redownload: Flag to force re-download and extraction of PPG files.
        """
        if os.path.exists(self.ppgpath) and not redownload:
            print("PPG files already exist")
            return

        link = "https://figshare.com/ndownloader/articles/1209662/versions/6"
        print("Downloading PPG files (2.47 GB) ...")
        self.download_file(link, self.zippath)
        
        print("Unzipping PPG files ...")
        with zipfile.ZipFile(self.zippath, "r") as zip_ref:
            zip_ref.extractall(self.ppgpath)
    
        os.remove(self.zippath)
        print("Done extracting and downloading")

    def download_file(self, url: str, filename: str) -> str:
        """
        Downloads a file from a URL to a specified local filename.

        :param url: URL to download the file from.
        :param filename: Local file path to save the downloaded file.
        :return: Path to the downloaded file.
        """
        chunk_size = 1024  # Define the size of data chunks to download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an error for bad responses
            total = int(r.headers.get("Content-Length", 0))  # Get the total file size
            with open(filename, "wb") as f, tqdm(unit="B", total=total) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)  # Write each chunk to the file
                    pbar.update(len(chunk))  # Update the progress bar
        return filename



    def process_data(self):
        chunk_size = int(self.fs_target * 10)

        train_X, train_y = [], []
        val_X, val_y = [], []
        test_X, test_y = [], []

        self.labels = pd.read_csv(os.path.join(self.ppgpath, "AHI.csv"))
        filenames = sorted([f for f in os.listdir(self.ppgpath) if f.startswith("subject")])
        for fname in tqdm(filenames):
            df = pd.read_csv(os.path.join(self.ppgpath, fname))
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
        np.save(os.path.join(self.ppgpath, f"train_X_ppg_{self.fs_target}Hz.npy"), train_X)
        np.save(os.path.join(self.ppgpath, f"train_y_sdb.npy"), train_y)

        np.save(os.path.join(self.ppgpath, f"val_X_ppg_{self.fs_target}Hz.npy"), val_X)
        np.save(os.path.join(self.ppgpath, f"val_y_sdb.npy"), val_y)

        np.save(os.path.join(self.ppgpath, f"test_X_ppg_{self.fs_target}Hz.npy"), test_X)
        np.save(os.path.join(self.ppgpath, f"test_y_sdb.npy"), test_y)


def main(newhz: int, zippath: str, ppgpath: str):
    processor = SDBDataProcessor(zippath=zippath, ppgpath=ppgpath, fs_target=newhz)
    processor.downloadextract_PPGfiles()
    processor.process_data()
    print(f"PPGBP PPG data files for binary classification are ready in {os.path.abspath(processor.ppgpath)}")

if __name__ == "__main__":
    main(newhz=50, zippath="../ppg.zip", ppgpath="../data/sdb")
    main(newhz=125, zippath="../ppg.zip", ppgpath="../data/sdb")
