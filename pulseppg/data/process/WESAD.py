import os
import pickle
import numpy as np
from typing import List, Tuple
import requests
from tqdm import tqdm
import zipfile
from scipy.signal import butter, lfilter
from scipy import fftpack
from utils import resample_lerp_vectorized as resample_lerp, resample_batch_signal

class BasePPGProcessor:
    """
    Base class for processing Photoplethysmography (PPG) data. Handles downloading, extracting, 
    and processing of PPG files.
    """
    def __init__(self, zippath: str, ppgpath: str, processedppgpath: str, newhz: int):
        """
        Initializes the processor with paths for zip, raw, and processed PPG data, and the target frequency.
        
        :param zippath: Path to the zip file for downloading PPG data.
        :param ppgpath: Path to store the extracted raw PPG data.
        :param processedppgpath: Path to store the processed PPG data.
        :param newhz: New sampling frequency for processing.
        """
        self.zippath = zippath
        self.ppgpath = ppgpath
        self.processedppgpath = processedppgpath
        self.newhz = newhz

    def downloadextract_PPGfiles(self, redownload: bool = False) -> None:
        """
        Downloads and extracts PPG files if they do not already exist or if redownload is requested.

        :param redownload: Flag to force re-download and extraction of PPG files.
        """
        if os.path.exists(self.ppgpath) and not redownload:
            print("PPG files already exist")
            return

        link = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
        print("Downloading PPG files (2.5 GB) ...")
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
        
    # original paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9358140
    # code from here: https://github.com/seongsilheo/stress_classification_with_PPG/blob/master/preprocessing_tool/noise_reduction.py
    
    def denoisePPG(self, ppg_input: np.ndarray, orighz: int = 64) -> np.ndarray:
        """
        Denoises PPG input data using bandpass filtering and Fourier reconstruction.

        :param ppg_input: PPG signal input array.
        :param orighz: Original sampling frequency of the PPG signal.
        :return: Denoised PPG signal.
        """
        ppg_bp = self.butter_bandpassfilter(ppg_input, 0.5, 10, orighz, order=2)
        signal_one_percent = int(len(ppg_bp))
        cutoff = self.get_cutoff(ppg_bp[:signal_one_percent], orighz)
        sec = 12
        N = orighz * sec
        overlap = int(np.round(N * 0.02))
        ppg_freq = self.compute_and_reconstruction_dft(ppg_bp, orighz, sec, overlap, cutoff)

        # Apply moving average for smoothing
        fwd = self.movingaverage(ppg_freq, size=3)
        bwd = self.movingaverage(ppg_freq[::-1], size=3)
        ppg_ma = np.mean(np.vstack((fwd, bwd[::-1])), axis=0)

        ppg_real = np.real(ppg_ma)

        # Resample to the new frequency
        if self.newhz < orighz:
            ppg_newhz = resample_lerp(ppg_real, orighz=orighz, newhz=self.newhz)
        else:
            ppg_newhz = resample_batch_signal(ppg_real, orighz=orighz, newhz=self.newhz)

        return self.znorm_percent(ppg_newhz, percent=90)

    def znorm_percent(self, signal: np.ndarray, percent: int = 90) -> np.ndarray:
        """
        Normalizes the signal using z-normalization up to a specified percentile.

        :param signal: Input signal array.
        :param percent: Percentile threshold for normalization.
        :return: Normalized signal.
        """
        signal_passpercent = signal[signal < np.percentile(signal, percent)]
        mean = np.mean(signal_passpercent)
        stddev = np.std(signal_passpercent)
        return (signal - mean) / stddev

    def butter_bandpass(self, lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Designs a Butterworth bandpass filter.

        :param lowcut: Lower frequency cutoff.
        :param highcut: Upper frequency cutoff.
        :param fs: Sampling frequency.
        :param order: Filter order.
        :return: Filter coefficients (b, a).
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def movingaverage(self, data: np.ndarray, size: int = 4) -> np.ndarray:
        """
        Computes the moving average of the data.

        :param Input data array.
        :param size: Size of the moving average window.
        :return: Smoothed data array.
        """
        data_set = np.asarray(data)
        weights = np.ones(size) / size
        return np.convolve(data_set, weights, mode='valid')

    def get_cutoff(self, block: np.ndarray, fs: int) -> List[float]:
        """
        Determines cutoff frequencies for filtering based on the input signal block.

        :param block: Input signal block.
        :param fs: Sampling frequency.
        :return: List of low and high cutoff frequencies.
        """
        block = np.array([item.real for item in block])
        peak = self.threshold_peakdetection(block, fs)
        hr_mean = np.mean(self.calc_heartrate(self.RR_interval(peak, fs)))
        low_cutoff = np.round(hr_mean / 60 - 0.6, 1)
        frequencies, fourierTransform, timePeriod = self.FFT(block, fs)
        ths = max(abs(fourierTransform)) * 0.1

        for i in range(int(5 * timePeriod), 0, -1):
            if abs(fourierTransform[i]) > ths:
                high_cutoff = np.round(i / timePeriod, 1)
                break
        return [low_cutoff, high_cutoff]

    def calc_heartrate(self, RR_list: List[float]) -> List[float]:
        """
        Calculates heart rates from RR intervals.

        :param RR_list: List of RR intervals.
        :return: List of heart rates.
        """
        HR = []
        window_size = 10
        for val in RR_list:
            if 400 < val < 1500:
                heart_rate = 60000.0 / val
            elif (0 < val < 400) or val > 1500:
                heart_rate = np.mean(HR[-window_size:]) if len(HR) > 0 else 60.0
            else:
                heart_rate = 0.0
            HR.append(heart_rate)
        return HR

    def threshold_peakdetection(self, dataset: np.ndarray, fs: int) -> List[int]:
        """
        Detects peaks in the dataset using a threshold method.

        :param dataset: Input data array.
        :param fs: Sampling frequency.
        :return: List of indices where peaks are detected.
        """
        window = []
        peaklist = []
        listpos = 0
        localaverage = np.average(dataset)
        TH_elapsed = np.ceil(0.36 * fs)
        npeaks = 0

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
        
        peakarray = []
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

    def compute_and_reconstruction_dft(self, data: np.ndarray, fs: int, sec: int, overlap: int, cutoff: List[float]) -> np.ndarray:
        """
        Performs DFT and reconstructs the signal with specified cutoff frequencies.

        :param Input data array.
        :param fs: Sampling frequency.
        :param sec: Seconds for each segment.
        :param overlap: Overlap size between segments.
        :param cutoff: Low and high cutoff frequencies.
        :return: Reconstructed signal.
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
        Computes RR intervals from detected peaks.

        :param peaklist: List of detected peak indices.
        :param fs: Sampling frequency.
        :return: List of RR intervals in milliseconds.
        """
        RR_list = []
        for cnt in range(len(peaklist) - 1):
            RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
            ms_dist = (RR_interval / fs) * 1000.0
            RR_list.append(ms_dist)
        return RR_list

    def FFT(self, block: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Performs FFT on a signal block and returns frequency domain representation.

        :param block: Input signal block.
        :param fs: Sampling frequency.
        :return: Frequencies, Fourier transform, and time period of the block.
        """
        fourierTransform = np.fft.fft(block) / len(block)
        fourierTransform = fourierTransform[range(int(len(block) / 2))]
        tpCount = len(block)
        values = np.arange(int(tpCount) / 2)
        timePeriod = tpCount / fs
        frequencies = values / timePeriod
        return frequencies, fourierTransform, timePeriod

    def butter_bandpassfilter(self, data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter to the data.

        :param Input data array.
        :param lowcut: Lower cutoff frequency.
        :param highcut: Upper cutoff frequency.
        :param fs: Sampling frequency.
        :param order: Order of the filter.
        :return: Filtered data.
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data, axis=0)
        return y

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Loads PPG data from files into memory.

        :return: Tuple of PPG signals, labels, and patient names.
        """
        print("Processing WESAD PPG...")
        ppgs, labels, names = [], [], []
        folders = os.listdir(os.path.join(self.ppgpath, "WESAD"))
        folders.sort()

        for patient in folders:
            if patient[0] != "S":
                continue
            names.append(patient)
            with open(os.path.join(self.ppgpath, "WESAD", f"{patient}/{patient}.pkl"), "rb") as f:
                patientfile = pickle.load(f, encoding='latin1')
                ppgs.append(patientfile["signal"]["wrist"]["BVP"])
                labels.append(patientfile["label"])

        return ppgs, labels, np.array(names)

    def truncate_data(self, ppgs: List[np.ndarray], labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Truncates the data to the minimum length among all samples.

        :param ppgs: List of PPG signal arrays.
        :param labels: List of label arrays corresponding to PPG signals.
        :return: Truncated PPG and label arrays.
        """
        minlengthofppg = min(len(ppg) for ppg in ppgs)
        minlengthoflabels = min(len(label) for label in labels)

        ppgs_minlen = np.array([ppg[:minlengthofppg] for ppg in ppgs])
        labels_minlen = np.array([label[:minlengthoflabels] for label in labels])

        return ppgs_minlen, labels_minlen

    def denoise_all(self, ppgs_minlen: np.ndarray) -> np.ndarray:
        """
        Applies denoising to all PPG signals.

        :param ppgs_minlen: Array of truncated PPG signals.
        :return: Array of denoised PPG signals.
        """
        print("Denoising WESAD PPG ...")
        ppgs_filtered = []
        for i in range(ppgs_minlen.shape[0]):
            ppg_filtered = self.denoisePPG(ppgs_minlen[i, :, 0])
            ppgs_filtered.append(ppg_filtered)

        return np.expand_dims(np.array(ppgs_filtered), 2)

    def create_splits(self, ppgs_filtered: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates train, validation, and test splits from the denoised PPG data.

        :param ppgs_filtered: Array of denoised PPG signals.
        :return: Indices for train, validation, and test data splits.
        """
        np.random.seed(1234)
        inds = np.arange(ppgs_filtered.shape[0]).astype(int)
        np.random.shuffle(inds)

        os.makedirs(self.processedppgpath, exist_ok=True)

        train_inds = inds[:11]
        val_inds = inds[11:13]
        test_inds = inds[13:15]

        return train_inds, val_inds, test_inds
                
    def preprocess_PPGdata(self) -> None:
        """
        Preprocesses PPG data for multi-class classification.
        """
        ppgs, labels, names = self.load_data()
        ppgs_minlen, labels_minlen = self.truncate_data(ppgs, labels)
        labels_original = labels
        ppgs_filtered = self.denoise_all(ppgs_minlen)
        train_inds, val_inds, test_inds = self.create_splits(ppgs_filtered)
        self.create_subsequences_and_save(ppgs_filtered, labels_original, names, train_inds, val_inds, test_inds)        


    def save_data(self, data_subseq: List[np.ndarray], names_subseq: List[str], labels_subseq: List[np.ndarray], subset: str) -> None:
        """
        Saves the processed subsequences to disk.

        :param data_subseq: List of data subsequences.
        :param names_subseq: List of corresponding participant names.
        :param labels_subseq: List of corresponding labels.
        :param subset: Subset identifier (train, val, or test).
        """
        data_numpy = np.transpose(np.concatenate(data_subseq), (0, 2, 1))
        names_numpy = np.array(names_subseq)
        labels_numpy = np.concatenate(labels_subseq) - 1

        np.save(os.path.join(self.processedppgpath, f"{subset}_X_ppg_{self.newhz}Hz.npy"), data_numpy)
        np.save(os.path.join(self.processedppgpath, f"{subset}_names_subseq_{self.newhz}Hz.npy"), names_numpy)
        np.save(os.path.join(self.processedppgpath, f"{subset}_y_stress_{self.newhz}Hz.npy"), labels_numpy)

class BinaryPPGProcessor(BasePPGProcessor):
    """    
    Processor for binary classification of PPG data.    
    """

    def create_subsequences_and_save(self, ppgs_filtered: np.ndarray, labels_original: List[np.ndarray], names: np.ndarray, train_inds: np.ndarray, val_inds: np.ndarray, test_inds: np.ndarray) -> None:
        """
        Creates subsequences from the PPG data and saves them.

        :param ppgs_filtered: Array of denoised PPG signals.
        :param labels_original: Original labels array.
        :param names: Array of participant names.
        :param train_inds: Indices for training data.
        :param val_inds: Indices for validation data.
        :param test_inds: Indices for test data.
        """
        subseq_size_label = 700 * 60
        subseq_size_data = self.newhz * 60

        data_subseq_train, data_subseq_val, data_subseq_test = [], [], []
        labels_subseq_train, labels_subseq_val, labels_subseq_test = [], [], []
        names_subseq_train, names_subseq_val, names_subseq_test = [], [], []

        T_ppg = ppgs_filtered.shape[1]
        for patient, label in enumerate(labels_original):
            uniques, uniques_index = np.unique(label, return_index=True)

            for unique, unique_startidx in zip(uniques, uniques_index):
                if unique not in [1, 2, 3]:
                    continue

                while True:
                    if unique_startidx // subseq_size_label * subseq_size_data > T_ppg:
                        break
                    for next_idx in range(unique_startidx, len(label)):
                        if unique != label[next_idx]:
                            break

                    totalsubseqs = (next_idx - unique_startidx) // subseq_size_label
                    startidx = unique_startidx // subseq_size_label * subseq_size_data
                    data_temp_60sec = ppgs_filtered[patient, startidx:startidx + totalsubseqs * subseq_size_data]

                    if unique_startidx // subseq_size_label * subseq_size_data + totalsubseqs * subseq_size_data > T_ppg:
                        totalsubseqs = data_temp_60sec.shape[0] // subseq_size_data
                        data_temp_60sec = data_temp_60sec[:totalsubseqs * subseq_size_data, :]

                    if totalsubseqs == 0:
                        break

                    data_temp_60sec = np.stack(np.split(data_temp_60sec, totalsubseqs, 0), 0)
                    unique_temp = 1 if unique == 3 else unique
                    label_temp_60sec = np.repeat(unique_temp, totalsubseqs)

                    if patient in train_inds:
                        data_subseq_train.append(data_temp_60sec)
                        labels_subseq_train.append(label_temp_60sec)
                        names_subseq_train.append(names[patient])
                    elif patient in val_inds:
                        data_subseq_val.append(data_temp_60sec)
                        labels_subseq_val.append(label_temp_60sec)
                        names_subseq_val.append(names[patient])
                    elif patient in test_inds:
                        data_subseq_test.append(data_temp_60sec)
                        labels_subseq_test.append(label_temp_60sec)
                        names_subseq_test.append(names[patient])
                    else:
                        import sys; sys.exit()

                    if unique != 4:
                        break

        self.save_data(data_subseq_train, names_subseq_train, labels_subseq_train, "train")
        self.save_data(data_subseq_val, names_subseq_val, labels_subseq_val, "val")
        self.save_data(data_subseq_test, names_subseq_test, labels_subseq_test, "test")


class MultiClassPPGProcessor(BasePPGProcessor):
    """
    Processor for multi-class classification of PPG data.    
    """

    def create_subsequences_and_save(self, ppgs_filtered: np.ndarray, labels_original: List[np.ndarray], names: np.ndarray, train_inds: np.ndarray, val_inds: np.ndarray, test_inds: np.ndarray) -> None:
        """
        Creates subsequences from the PPG data and saves them for multi-class classification.

        :param ppgs_filtered: Array of denoised PPG signals.
        :param labels_original: Original labels array.
        :param names: Array of participant names.
        :param train_inds: Indices for training data.
        :param val_inds: Indices for validation data.
        :param test_inds: Indices for test data.
        """
        subseq_size_label = 700 * 60
        subseq_size_data = self.newhz * 60

        data_subseq_train, data_subseq_val, data_subseq_test = [], [], []
        labels_subseq_train, labels_subseq_val, labels_subseq_test = [], [], []
        names_subseq_train, names_subseq_val, names_subseq_test = [], [], []

        T_ppg = ppgs_filtered.shape[1]
        for patient, label in enumerate(labels_original):
            uniques, uniques_index = np.unique(label, return_index=True)

            for unique, unique_startidx in zip(uniques, uniques_index):
                flag = False
                if unique not in [1, 2, 3, 4]:
                    continue
                
                while True:
                    if unique_startidx // subseq_size_label * subseq_size_data > T_ppg:
                        break
                    for next_idx in range(unique_startidx, len(label)):
                        if unique != label[next_idx]:
                            break

                    totalsubseqs = (next_idx - unique_startidx) // subseq_size_label
                    startidx = unique_startidx // subseq_size_label * subseq_size_data
                    data_temp_60sec = ppgs_filtered[patient, startidx:startidx + totalsubseqs * subseq_size_data]

                    if unique_startidx // subseq_size_label * subseq_size_data + totalsubseqs * subseq_size_data > T_ppg:
                        totalsubseqs = data_temp_60sec.shape[0] // subseq_size_data
                        data_temp_60sec = data_temp_60sec[:totalsubseqs * subseq_size_data, :]

                    if totalsubseqs == 0:
                        break

                    data_temp_60sec = np.stack(np.split(data_temp_60sec, totalsubseqs, 0), 0)
                    label_temp_60sec = np.repeat(unique, totalsubseqs)

                    if patient in train_inds:
                        data_subseq_train.append(data_temp_60sec)
                        labels_subseq_train.append(label_temp_60sec)
                        names_subseq_train.append(names[patient])
                    elif patient in val_inds:
                        data_subseq_val.append(data_temp_60sec)
                        labels_subseq_val.append(label_temp_60sec)
                        names_subseq_val.append(names[patient])
                    elif patient in test_inds:
                        data_subseq_test.append(data_temp_60sec)
                        labels_subseq_test.append(label_temp_60sec)
                        names_subseq_test.append(names[patient])
                    else:
                        import sys; sys.exit()
                        
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

        self.save_data(data_subseq_train, names_subseq_train, labels_subseq_train, "train")
        self.save_data(data_subseq_val, names_subseq_val, labels_subseq_val, "val")
        self.save_data(data_subseq_test, names_subseq_test, labels_subseq_test, "test")



def main(newhz: int) -> None:
    """
    Main function to process PPG data for both binary and multi-class classification.

    :param newhz: New sampling frequency for the data.
    """
    # Initialize and process binary classification data
    binary_processor = BinaryPPGProcessor(zippath="../ppg.zip", ppgpath="../ppg", processedppgpath="../ppg/binary", newhz=newhz)
    binary_processor.downloadextract_PPGfiles()
    binary_processor.preprocess_PPGdata()
    print(f"WESAD PPG data files for binary classification are ready in {os.path.abspath(binary_processor.processedppgpath)}") 

    # Initialize and process multi-class classification data
    multiclass_processor = MultiClassPPGProcessor(zippath="../ppg.zip", ppgpath="../ppg", processedppgpath="../ppg/multiclass", newhz=newhz)
    multiclass_processor.downloadextract_PPGfiles()
    multiclass_processor.preprocess_PPGdata()
    print(f"WESAD PPG data files for multiclass classification are ready in {os.path.abspath(multiclass_processor.processedppgpath)}") 

if __name__ == "__main__":
    main(newhz=50)