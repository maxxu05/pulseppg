"""
PPG Data Preprocessing and Denoising Module

This module handles downloading, extracting, and processing PPG (Photoplethysmogram) data from the 
WESAD dataset. It includes functionalities for denoising PPG signals using a bandpass filter, 
FFT-based reconstruction, and resampling. The module also processes accelerometer data associated 
with the PPG signals and prepares the data for machine learning tasks by creating subsequences.

Key Functions:
- downloadextract_PPGfiles: Downloads and extracts PPG data from a specified URL.
- preprocess_PPGProcesses PPG and accelerometer data, denoises signals, and creates subsequences.
- download_file: Utility function for downloading files with a progress bar.
- denoisePPG: Callable class that implements denoising methods for PPG signals.
"""

import os
import pickle
from typing import List, Tuple, Optional

import requests
from tqdm import tqdm
import zipfile
import numpy as np
from scipy.signal import butter, lfilter
from scipy import fftpack

# Import resampling utility
from utils import resample_lerp_vectorized as resample_lerp


def main() -> None:
    """
    Main entry point for the script.
    Downloads and preprocesses PPG files.
    """
    downloadextract_PPGfiles()
    preprocess_PPGdata()


def downloadextract_PPGfiles(zippath: str = "data/ppg.zip", targetpath: str = "data/ppg", redownload: bool = False) -> None:
    """
    Download and extract PPG files from a specified URL.

    Args:
        zippath: Local path for the downloaded zip file.
        targetpath: Directory where the files will be extracted.
        redownload: If True, will re-download the files even if they exist.
    """
    if os.path.exists(targetpath) and not redownload:
        print("PPG files already exist")
        return

    link = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"

    print("Downloading PPG files (2.5 GB) ...")
    download_file(link, zippath)

    print("Unzipping PPG files ...")
    with zipfile.ZipFile(zippath, "r") as zip_ref:
        zip_ref.extractall(targetpath)
    os.remove(zippath)
    print("Done extracting and downloading")


def preprocess_PPGdata(ppgpath: str = "data/ppg", processedppgpath: str = "data/ppg/processed", reprocess: bool = False) -> None:
    """
    Preprocess PPG data and accelerometer data from the WESAD dataset.

    Args:
        ppgpath: Path to the folder containing the extracted WESAD data.
        processedppgpath: Directory for storing processed numpy files.
        reprocess: If True, will reprocess even if files already exist.
    """
    print("Processing PPG files ...")
    ppgs: List[np.ndarray] = []
    accels: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    names: List[str] = []
    
    folders = os.listdir(os.path.join(ppgpath, "WESAD"))
    folders.sort()
    counter = 0

    for patient in folders:
        if counter == 5:  # Limit processing to the first 5 patients for testing
            break
        if patient[0] != "S":  # Ignore non-patient files
            continue
        names.append(patient)
        with open(os.path.join(ppgpath, "WESAD", f"{patient}/{patient}.pkl"), "rb") as f:
            patientfile = pickle.load(f, encoding='latin1')
            ppgs.append(patientfile["signal"]["wrist"]["BVP"])
            accels.append(patientfile["signal"]["wrist"]["ACC"])
            labels.append(patientfile["label"])
    
    names = np.array(names)

    # Find minimum length of PPG, accelerometer, and labels
    minlengthofppg = min(len(ppg) for ppg in ppgs)
    minlengthofaccel = min(len(accel) for accel in accels)
    minlengthoflabels = min(len(label) for label in labels)

    # Truncate data to match minimum lengths
    ppgs_minlen = np.array([ppg[:minlengthofppg] for ppg in ppgs])
    accels_minlen = np.array([accel[:minlengthofaccel, :] for accel in accels])
    labels_minlen = np.array([label[:minlengthoflabels] for label in labels])

    labels_original = labels

    # Denoise PPG signals
    print("Denoising PPG ...")
    ppgs_filtered = []
    denoisePPGfunc = denoisePPG()

    for i in range(ppgs_minlen.shape[0]):
        ppg_filtered = denoisePPGfunc(ppgs_minlen[i, :, 0])
        ppgs_filtered.append(ppg_filtered)
    
    ppgs_filtered = np.expand_dims(np.array(ppgs_filtered), 2)

    # Denoise accelerometer data
    accels_filtered = []
    for i in range(ppgs_minlen.shape[0]):
        accel_temp = []
        for j in range(3):  # Assuming 3 axes of accelerometer data
            accels_temp_j = resample_lerp(accels_minlen[i, :, j], orighz=32, newhz=50)
            accel_temp.append(accels_temp_j)
        accel_temp_np = np.stack(accel_temp)
        accels_filtered.append(accel_temp_np)
    accels_filtered = np.stack(accels_filtered)

    # Create reproducible splits
    np.random.seed(1234)
    inds = np.arange(ppgs_filtered.shape[0]).astype(int)
    np.random.shuffle(inds)

    os.makedirs(processedppgpath, exist_ok=True)

    train_inds = inds[:11]
    val_inds = inds[11:13]
    test_inds = inds[13:15]

    # Preparing subsequences for training, validation, and testing
    data_subseq_train, data_subseq_val, data_subseq_test = [], [], []
    accel_subseq_train, accel_subseq_val, accel_subseq_test = [], [], []
    labels_subseq_train, labels_subseq_val, labels_subseq_test = [], [], []
    names_subseq_train, names_subseq_val, names_subseq_test = [], [], []

    subseq_size_label = 700 * 60  # 1 minute of labels at 700 Hz
    subseq_size_data = 50 * 60     # 1 minute of PPG at 50 Hz

    T_ppg = ppgs_filtered.shape[1]
    for patient, label in enumerate(labels_original):
        uniques, uniques_index = np.unique(label, return_index=True)

        for unique, unique_startidx in zip(uniques, uniques_index):
            flag = False
            if unique not in [1, 2, 3, 4]:  # Only keep certain labels
                continue
            
            while True:
                if unique_startidx // subseq_size_label * subseq_size_data > T_ppg:
                    break
                for next_idx in range(unique_startidx, len(label)):
                    if unique != label[next_idx]:
                        break

                # Break into 1 minute intervals
                totalsubseqs = (next_idx - unique_startidx) // subseq_size_label 
                startidx = unique_startidx // subseq_size_label * subseq_size_data
                data_temp_60sec = ppgs_filtered[patient, startidx:startidx + totalsubseqs * subseq_size_data]
                accel_temp_60sec = accels_filtered[patient, :, startidx:startidx + totalsubseqs * subseq_size_data]

                if unique_startidx // subseq_size_label * subseq_size_data + totalsubseqs * subseq_size_data > T_ppg:
                    totalsubseqs = data_temp_60sec.shape[0] // subseq_size_data
                    data_temp_60sec = data_temp_60sec[:totalsubseqs * subseq_size_data, :]
                    accel_temp_60sec = accel_temp_60sec[:, :totalsubseqs * subseq_size_data]

                if totalsubseqs == 0:  # Edge case handling
                    break
                
                data_temp_60sec = np.stack(np.split(data_temp_60sec, totalsubseqs, 0), 0)
                accel_temp_60sec = np.stack(np.split(accel_temp_60sec, totalsubseqs, -1), 0)
                label_temp_60sec = np.repeat(unique, totalsubseqs)

                assert data_temp_60sec.shape[1] == accel_temp_60sec.shape[-1]

                # Assign to respective subsets
                if patient in train_inds:
                    data_subseq_train.append(data_temp_60sec)
                    accel_subseq_train.append(accel_temp_60sec)
                    labels_subseq_train.append(label_temp_60sec)
                    names_subseq_train.append(names[patient])
                elif patient in val_inds:
                    data_subseq_val.append(data_temp_60sec)
                    accel_subseq_val.append(accel_temp_60sec)
                    labels_subseq_val.append(label_temp_60sec)
                    names_subseq_val.append(names[patient])
                elif patient in test_inds:
                    data_subseq_test.append(data_temp_60sec)
                    accel_subseq_test.append(accel_temp_60sec)
                    labels_subseq_test.append(label_temp_60sec)
                    names_subseq_test.append(names[patient])
                else:
                    import sys; sys.exit()  # Error in assignment

                if unique != 4:
                    break
                else:
                    if flag:
                        break
                    flag = True
                    newlabel = label[next_idx:]
                    uniques_temp, uniques_indedata_temp = np.unique(newlabel, return_index=True)
                    try:
                        unique_startidx = uniques_indedata_temp[np.where(uniques_temp == 4)][0] + next_idx
                    except IndexError:
                        break

    # Save processed data
    data_subseq_train_numpy = np.transpose(np.concatenate(data_subseq_train), (0, 2, 1))
    accel_subseq_train_numpy = np.concatenate(accel_subseq_train)
    names_subseq_train_numpy = np.array(names_subseq_train)
    labels_subseq_train_numpy = np.concatenate(labels_subseq_train) - 1

    np.save(os.path.join(processedppgpath, "train_X_ppg.npy"), data_subseq_train_numpy)
    np.save(os.path.join(processedppgpath, "train_X_accel.npy"), accel_subseq_train_numpy)
    np.save(os.path.join(processedppgpath, "train_names_subseq.npy"), names_subseq_train_numpy)
    np.save(os.path.join(processedppgpath, "train_y_stress.npy"), labels_subseq_train_numpy)

    data_subseq_val_numpy = np.transpose(np.concatenate(data_subseq_val), (0, 2, 1))
    accel_subseq_val_numpy = np.concatenate(accel_subseq_val)
    names_subseq_val_numpy = np.array(names_subseq_val)
    labels_subseq_val_numpy = np.concatenate(labels_subseq_val) - 1
    np.save(os.path.join(processedppgpath, "val_X_ppg.npy"), data_subseq_val_numpy)
    np.save(os.path.join(processedppgpath, "val_X_accel.npy"), accel_subseq_val_numpy)
    np.save(os.path.join(processedppgpath, "val_names_subseq.npy"), names_subseq_val_numpy)
    np.save(os.path.join(processedppgpath, "val_y_stress.npy"), labels_subseq_val_numpy)

    data_subseq_test_numpy = np.transpose(np.concatenate(data_subseq_test), (0, 2, 1))
    accel_subseq_test_numpy = np.concatenate(accel_subseq_test)
    names_subseq_test_numpy = np.array(names_subseq_test)
    labels_subseq_test_numpy = np.concatenate(labels_subseq_test) - 1
    np.save(os.path.join(processedppgpath, "test_X_ppg.npy"), data_subseq_test_numpy)
    np.save(os.path.join(processedppgpath, "test_X_accel.npy"), accel_subseq_test_numpy)
    np.save(os.path.join(processedppgpath, "test_names_subseq.npy"), names_subseq_test_numpy)
    np.save(os.path.join(processedppgpath, "test_y_stress.npy"), labels_subseq_test_numpy)


class denoisePPG:
    """
    Callable class for denoising PPG signals.

    Implements:
    - Bandpass filtering
    - FFT reconstruction
    - Moving average smoothing
    - Resampling
    - Z-normalization based on percentile
    """

    def __call__(self, ppg_input: np.ndarray, orighz: int = 64, newhz: int = 50) -> np.ndarray:
        """
        Denoise and resample a single-channel PPG signal.

        Args:
            ppg_input: 1D numpy array of PPG signal.
            orighz: Original sampling frequency (Hz).
            newhz: Desired sampling frequency (Hz).

        Returns:
            Denoised and resampled PPG signal.
        """
        ppg_bp = self.butter_bandpassfilter(ppg_input, 0.5, 10, orighz, order=2)
        signal_one_percent = int(len(ppg_bp))
        cutoff = self.get_cutoff(ppg_bp[:signal_one_percent], orighz)
        sec = 12
        N = orighz * sec
        overlap = int(np.round(N * 0.02))
        ppg_freq = self.compute_and_reconstruction_dft(ppg_bp, orighz, sec, overlap, cutoff)

        # Apply moving average smoothing
        fwd = self.movingaverage(ppg_freq, size=3)
        bwd = self.movingaverage(ppg_freq[::-1], size=3)
        ppg_ma = np.mean(np.vstack((fwd, bwd[::-1])), axis=0)

        ppg_real = np.real(ppg_ma)

        ppg_newhz = resample_lerp(ppg_real, orighz=orighz, newhz=newhz)
        ppg_znormed = self.znorm_percent(ppg_newhz, percent=90)

        return ppg_znormed

    def znorm_percent(self, signal: np.ndarray, percent: int = 90) -> np.ndarray:
        """
        Z-normalize the signal using mean and stddev calculated from values below a given percentile.

        Args:
            signal: 1D numpy array of PPG signal.
            percent: Percentile to use for normalization.

        Returns:
            Z-normalized signal.
        """
        signal_passpercent = signal[signal < np.percentile(signal, percent)]
        mean = np.mean(signal_passpercent)
        stddev = np.std(signal_passpercent)

        return (signal - mean) / stddev

    def butter_bandpass(self, lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def movingaverage(self, np.ndarray, size: int = 4) -> np.ndarray:
        """
        Compute moving average over the data.

        Args:
            Input array.
            size: Window size for moving average.

        Returns:
            Moving average of the data.
        """
        result = []
        data_set = np.asarray(data)
        weights = np.ones(size) / size
        result = np.convolve(data_set, weights, mode='valid')
        return result

    def get_cutoff(self, block: np.ndarray, fs: int) -> List[float]:
        """
        Get dynamic cutoff frequencies for FFT filtering based on signal characteristics.

        Args:
            block: 1D numpy array of PPG signal to analyze.
            fs: Sampling frequency.

        Returns:
            List of low and high cutoff frequencies.
        """
        block = np.array([item.real for item in block])
        peak = self.threshold_peakdetection(block, fs)
        hr_mean = np.mean(self.calc_heartrate(self.RR_interval(peak, fs)))
        low_cutoff = np.round(hr_mean / 60 - 0.6, 1)  # 0.6
        frequencies, fourierTransform, timePeriod = self.FFT(block, fs)
        ths = max(abs(fourierTransform)) * 0.1  # threshold for filtering

        for i in range(int(5 * timePeriod), 0, -1):  # check from 5th harmonic
            if abs(fourierTransform[i]) > ths:
                high_cutoff = np.round(i / timePeriod, 1)
                break
        return [low_cutoff, high_cutoff]

    def calc_heartrate(self, RR_list: List[float]) -> List[float]:
        """
        Calculate heart rate from RR intervals.

        Args:
            RR_list: List of RR intervals in milliseconds.

        Returns:
            List of calculated heart rates.
        """
        HR = []
        window_size = 10
        for val in RR_list:
            if 400 < val < 1500:
                heart_rate = 60000.0 / val  # 60000 ms in a minute
            elif (0 < val < 400) or val > 1500:
                heart_rate = np.mean(HR[-window_size:]) if len(HR) > 0 else 60.0
            else:
                heart_rate = 0.0
            HR.append(heart_rate)
        return HR

    def threshold_peakdetection(self, dataset: np.ndarray, fs: int) -> List[int]:
        """
        Detect peaks in the dataset using a threshold method.

        Args:
            dataset: 1D array of PPG signal.
            fs: Sampling frequency.

        Returns:
            List of detected peak positions.
        """
        window = []
        peaklist = []
        listpos = 0
        localaverage = np.average(dataset)
        TH_elapsed = np.ceil(0.36 * fs)
        npeaks = 0
        peakarray = []

        for datapoint in dataset:
            if (datapoint < localaverage) and (len(window) < 1):
                listpos += 1
            elif (datapoint >= localaverage):
                window.append(datapoint)
                listpos += 1
            else:
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(maximum))
                peaklist.append(beatposition)
                window = []
                listpos += 1
        
        # Enforce minimum distance between peaks
        for val in peaklist:
            if npeaks > 0:
                prev_peak = peaklist[npeaks - 1]
                elapsed = val - prev_peak
                if elapsed > TH_elapsed:
                    peakarray.append(val)
            else:
                peakarray.append(val)
            npeaks += 1

        return peaklist

    def compute_and_reconstruction_dft(self, np.ndarray, fs: int, sec: int, overlap: int, cutoff: List[float]) -> np.ndarray:
        """
        Perform block-wise FFT filtering and reconstruct the signal.

        Args:
            Input signal data.
            fs: Sampling frequency.
            sec: Block size in seconds.
            overlap: Overlap between blocks.
            cutoff: Frequency cutoff for filtering.

        Returns:
            Reconstructed signal after filtering.
        """
        concatenated_sig = []
        for i in range(0, len(data), fs * sec - overlap):
            seg_data = data[i:i + fs * sec]
            sig_fft = fftpack.fft(seg_data)
            sample_freq = (fftpack.fftfreq(len(seg_data)) * fs)
            new_freq_fft = sig_fft.copy()
            new_freq_fft[np.abs(sample_freq) < cutoff[0]] = 0
            new_freq_fft[np.abs(sample_freq) > cutoff[1]] = 0
            filtered_sig = fftpack.ifft(new_freq_fft)

            if i == 0:
                concatenated_sig = np.hstack([concatenated_sig, filtered_sig[:fs * sec - overlap // 2]])
            elif i == len(data) - 1:
                concatenated_sig = np.hstack([concatenated_sig, filtered_sig[overlap // 2:]])
            else:
                concatenated_sig = np.hstack([concatenated_sig, filtered_sig[overlap // 2:fs * sec - overlap // 2]])
        
        return concatenated_sig

    def RR_interval(self, peaklist: List[int], fs: int) -> List[float]:
        """
        Calculate RR intervals from detected peaks.

        Args:
            peaklist: List of detected peak positions.
            fs: Sampling frequency.

        Returns:
            List of RR intervals in milliseconds.
        """
        RR_list = []
        for cnt in range(len(peaklist) - 1):
            RR_interval = (peaklist[cnt + 1] - peaklist[cnt])  # Distance between beats in samples
            ms_dist = (RR_interval / fs) * 1000.0  # Convert to milliseconds
            RR_list.append(ms_dist)
        return RR_list

    def FFT(self, block: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute FFT of the block and return frequency components.

        Args:
            block: Input signal block.
            fs: Sampling frequency.

        Returns:
            Frequencies, FFT of the block, and time period.
        """
        fourierTransform = np.fft.fft(block) / len(block)  # Normalize
        fourierTransform = fourierTransform[range(int(len(block) / 2))]  # Single side frequency
        tpCount = len(block)
        values = np.arange(int(tpCount) / 2)
        timePeriod = tpCount / fs
        frequencies = values / timePeriod  # Frequency components
        return frequencies, fourierTransform, timePeriod

    def butter_bandpassfilter(self, np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter to the data.

        Args:
            Input signal.
            lowcut: Low cutoff frequency.
            highcut: High cutoff frequency.
            fs: Sampling frequency.
            order: Filter order.

        Returns:
            Filtered signal.
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data, axis=0)
        return y


def download_file(url: str, filename: str) -> str:
    """
    Download a file from the specified URL to the given local filename.

    Args:
        url: URL of the file to download.
        filename: Local path to save the downloaded file.

    Returns:
        The filename of the downloaded file.
    """
    chunk_size = 1024
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(filename, "wb") as f, tqdm(unit="B", total=total) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    return filename


if __name__ == "__main__":
    main()