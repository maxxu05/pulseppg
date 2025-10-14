"""
This function constructs a dummy dataset that mimics our original pre-training dataset. 
Due to privacy concerns and IRB restrictions, we will be unable to release our pre-training 
data. Therefore, if you would like to pre-train the PulsePPG model on 
your own dataset, please pre-process the data so that the numpy time-series files follows 
the below structure.

pulseppg/
└── data/
    └── datasets/
        └── dummydataset/
            ├── train/
            │   ├── subject_0/
            │   │   ├── hour_0/
            │   │   │   ├── ts_0.npy
            │   │   │   ├── ts_1.npy
            │   │   │   └── ...
            │   │   └── hour_1/
            │   │       ├── ts_0.npy
            │   │       ├── ts_1.npy
            │   │       └── ...
            │   └── subject_1/
            │       └── ...
            ├── val/
            │   ├── subject_32/
            │   │   └── hour_0/
            │   │       ├── ts_0.npy
            │   │       └── ...
            │   └── subject_33/
            │       └── ...
            └── test/
                ├── subject_48/
                │   └── hour_0/
                │       ├── ts_0.npy
                │       └── ...
                └── subject_49/
                    └── ...

"""

import numpy as np
import os
from tqdm import tqdm

PATH = "pulseppg/data/datasets/dummydataset"
NUM_SUBJECTS = 64
NUM_HOURS_PER_SUBJECT = 2
CHANNELS = 1 

#### CHANGE TIMELEN TO YOUR OWN TIME LENGTH ####
# PulsePPG will work with varying time lengths
TIMELEN = 12000  # 4min long sequence sampled at 50 Hz
NUM_TS_PER_HOUR = 15


def main():
    os.makedirs(PATH, exist_ok=True)
    for subject_id in tqdm(range(NUM_SUBJECTS)):
        # construct parent folder for train/val/test
        if subject_id < NUM_SUBJECTS // 2:
            TYPE = "train"
        elif subject_id < 3 * NUM_SUBJECTS // 4:
            TYPE = "val"
        else:
            TYPE = "test"
        typepath = os.path.join(PATH, TYPE)
        os.makedirs(typepath, exist_ok=True)

        # construct sub-parent folder for the subject-level
        subjectpath = os.path.join(typepath, f"subject_{subject_id}")
        os.makedirs(subjectpath, exist_ok=True)
        for hour_id in range(NUM_HOURS_PER_SUBJECT):
            # Construct child folder for hour-level
            # The other time-series alongside the anchor time-series within the folder form the "within-user" candidates
            hourpath = os.path.join(subjectpath, f"hour_{hour_id}")
            os.makedirs(hourpath, exist_ok=True)

            # construct 4 minute chunks sampled at 50hz
            for i in range(NUM_TS_PER_HOUR):
                # models expect numpy arrays of size TIMELEN, CHANNELS
                timeseries = np.random.normal(size=(TIMELEN, CHANNELS))
                np.save(os.path.join(hourpath, f"ts_{i}"), timeseries)


if __name__ == "__main__":
    main()