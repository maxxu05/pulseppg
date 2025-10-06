import os
import pandas as pd
import numpy as np
import zipfile
import requests
from tqdm import tqdm
from torch_ecg._preprocessors import Normalize
from utils import resample_batch_signal, preprocess_one_ppg_signal
import joblib

class PPGDataProcessor:
    def __init__(self, zippath: str, ppgpath: str, fs_target: int):
        # self.ppgpath = download_dir
        self.zippath = zippath
        self.ppgpath = ppgpath

        self.fs_target = fs_target
        self.fs = 1000  # Original frequency
        self.norm = Normalize(method='z-score')
        self.df = None

        ## this was added in order to match papagei's original splits
        self.train_ids = [2, 6, 8, 10, 12, 15, 16, 17, 18, 19, 22, 23, 26, 31, 32, 34, 35, 38, 40, 45, 48, 50, 53, 55, 56, 58, 60, 61, 63, 65, 66, 83, 85, 87, 89, 92, 93, 97, 98, 99, 100, 104, 105, 106, 107, 112, 113, 114, 116, 120, 122, 126, 128, 131, 134, 135, 137, 138, 139, 140, 141, 146, 148, 149, 152, 153, 154, 158, 160, 162, 164, 165, 167, 169, 170, 175, 176, 179, 183, 184, 186, 188, 189, 190, 191, 193, 196, 197, 199, 205, 206, 207, 209, 210, 212, 216, 217, 218, 223, 226, 227, 230, 231, 233, 234, 240, 242, 243, 244, 246, 247, 248, 256, 257, 404, 407, 409, 412, 414, 415, 416, 417, 419]
        self.test_ids = [14, 21, 25, 51, 52, 62, 67, 86, 90, 96, 103, 108, 110, 119, 123, 124, 130, 142, 144, 157, 172, 173, 174, 180, 182, 185, 192, 195, 200, 201, 211, 214, 219, 221, 228, 239, 250, 403, 405, 406, 410]
        self.val_ids = [3, 11, 24, 27, 29, 30, 41, 43, 47, 64, 88, 91, 95, 115, 125, 127, 136, 145, 155, 156, 161, 163, 166, 178, 198, 203, 208, 213, 215, 222, 229, 232, 235, 237, 241, 245, 252, 254, 259, 411, 418]

    def downloadextract_PPGfiles(self, redownload: bool = False) -> None:
        """
        Downloads and extracts PPG files if they do not already exist or if redownload is requested.

        :param redownload: Flag to force re-download and extraction of PPG files.
        """
        if os.path.exists(self.ppgpath) and not redownload:
            print("PPG files already exist")
            return

        link = "https://figshare.com/ndownloader/articles/5459299/versions/5"
        print("Downloading PPG files (2.33 MB) ...")
        self.download_file(link, self.zippath)
        
        print("Unzipping PPG files ...")
        with zipfile.ZipFile(self.zippath, "r") as zip_ref:
            zip_ref.extractall(self.ppgpath)

        zip_path_main = os.path.join(self.ppgpath, "PPG-BP Database.zip")
        with zipfile.ZipFile(zip_path_main, "r") as zip_ref:
            zip_ref.extractall(self.ppgpath)        

            
        os.remove(self.zippath)
        os.remove(zip_path_main)
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
        self.df = pd.read_excel(f"{self.ppgpath}/Data File/PPG-BP dataset.xlsx", header=1)
        subjects = self.df.subject_ID.values
        main_dir = f"{self.ppgpath}/Data File/0_subject/"
        ppg_dir = f"{self.ppgpath}/Data File/ppg/"

        if not os.path.exists(ppg_dir):
            os.mkdir(ppg_dir)

        filenames = [f.split("_")[0] for f in os.listdir(main_dir)]

        for f in tqdm(filenames):
            segments = []
            for s in range(1, 4):
                print(f"Processing: {f}_{s}")
                signal = pd.read_csv(f"{main_dir}{f}_{str(s)}.txt", sep='\t', header=None)
                signal = signal.values.squeeze()[:-1]

                # Apply normalization
                normalized_signal, _ = self.norm.apply(signal, fs=self.fs)
                signal, _, _, _ = preprocess_one_ppg_signal(waveform=normalized_signal, frequency=self.fs)
                resampled_signal = resample_batch_signal(signal, fs_original=self.fs, fs_target=self.fs_target, axis=0)

                padding_needed = 10 * self.fs_target - len(resampled_signal)
                pad_left = padding_needed // 2
                pad_right = padding_needed - pad_left

                padded_signal = np.pad(resampled_signal, pad_width=(pad_left, pad_right))
                segments.append(padded_signal)

            segments = np.vstack(segments)
            child_dir = f.zfill(4)
            self.save_segments_to_directory(ppg_dir, child_dir, segments)

        self.split_and_save_labels()
        self.process_additional_data(ppg_dir)

    def save_segments_to_directory(self, save_dir: str, dir_name: str, segments: np.ndarray):
        subject_dir = os.path.join(save_dir, dir_name)
        os.makedirs(subject_dir, exist_ok=True)
        for i, segment in enumerate(segments):
            joblib.dump(segment, os.path.join(subject_dir, f'{i}.p'))

    def split_and_save_labels(self):
        self.df = self.df.rename(columns={
            "Sex(M/F)": "sex",
            "Age(year)": "age",
            "Systolic Blood Pressure(mmHg)": "sysbp",
            "Diastolic Blood Pressure(mmHg)": "diasbp",
            "Heart Rate(b/m)": "hr",
            "BMI(kg/m^2)": "bmi"
        }).fillna(0)
        self.df['Hypertension_Code'] = pd.factorize(self.df['Hypertension'])[0]

        df_train = self.df[self.df.subject_ID.isin(self.train_ids)]
        df_val = self.df[self.df.subject_ID.isin(self.val_ids)]
        df_test = self.df[self.df.subject_ID.isin(self.test_ids)]

        df_train.to_csv(f"{self.ppgpath}/Data File/train.csv", index=False)
        df_val.to_csv(f"{self.ppgpath}/Data File/val.csv", index=False)
        df_test.to_csv(f"{self.ppgpath}/Data File/test.csv", index=False)

    def process_additional_data(self, ppg_dir):
        for name, ids in zip(["train", "test", "val"], [self.train_ids, self.test_ids, self.val_ids]):
            data, label_sysbp, label_diasbp, label_hr, label_ht = [], [], [], [], []
            label_averagesegmentamt, label_sysbp_patient, label_diasbp_patient, label_hr_patient, label_ht_patient = [], [], [], [], []

            for id_i in ids:
                label_averagesegmentamt_temp = 0
                for j in range(3):
                    signal = joblib.load(os.path.join(ppg_dir, f"{id_i:04}", f'{j}.p'))[None, None, :]
                    data.append(signal)

                    row = self.df[self.df["subject_ID"] == id_i]
                    if row.empty:
                        print(f"No data found for subject_ID {id_i}")
                        continue

                    label_sysbp.append(row["sysbp"].values[0])
                    label_diasbp.append(row["diasbp"].values[0])
                    label_hr.append(row["hr"].values[0])
                    label_ht.append(row["Hypertension_Code"].values[0])
                    label_averagesegmentamt_temp += 1
                label_averagesegmentamt.append(label_averagesegmentamt_temp)
                label_sysbp_patient.append(row["sysbp"].values[0])
                label_diasbp_patient.append(row["diasbp"].values[0])
                label_hr_patient.append(row["hr"].values[0])
                label_ht_patient.append(row["Hypertension_Code"].values[0])

            data = np.concatenate(data)
            label_sysbp = np.array(label_sysbp)
            label_diasbp = np.array(label_diasbp)
            label_hr = np.array(label_hr)
            label_ht = np.array(label_ht)

            np.save(os.path.join(self.ppgpath, f"{name}_X_ppg_{self.fs_target}Hz"), data)
            np.save(os.path.join(self.ppgpath, f"{name}_y_sysbp"), label_sysbp)
            np.save(os.path.join(self.ppgpath, f"{name}_y_diasbp"), label_diasbp)
            np.save(os.path.join(self.ppgpath, f"{name}_y_hr"), label_hr)
            np.save(os.path.join(self.ppgpath, f"{name}_y_ht"), label_ht)

            label_averagesegmentamt = np.array(label_averagesegmentamt)
            label_sysbp_patient = np.array(label_sysbp_patient)
            label_diasbp_patient = np.array(label_diasbp_patient)
            label_hr_patient = np.array(label_hr_patient)
            label_ht_patient = np.array(label_ht_patient)

            np.save(os.path.join(self.ppgpath, f"{name}_avgsegamt"), label_averagesegmentamt)
            np.save(os.path.join(self.ppgpath, f"{name}_y_sysbp_patient"), label_sysbp_patient)
            np.save(os.path.join(self.ppgpath, f"{name}_y_diasbp_patient"), label_diasbp_patient)
            np.save(os.path.join(self.ppgpath, f"{name}_y_hr_patient"), label_hr_patient)
            np.save(os.path.join(self.ppgpath, f"{name}_y_ht_patient"), label_ht_patient)

def main(newhz: int, zippath: str, ppgpath: str):
    processor = PPGDataProcessor(zippath=zippath, ppgpath=ppgpath, fs_target=newhz)
    processor.downloadextract_PPGfiles()
    processor.process_data()
    print(f"PPGBP PPG data files for binary classification are ready in {os.path.abspath(processor.ppgpath)}")

if __name__ == "__main__":
    main(newhz=50, zippath="../ppg.zip", ppgpath="../data/ppgbp",)
    main(newhz=125, zippath="../ppg.zip", ppgpath="../data/ppgbp",)