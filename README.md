## Dataset Setup

This project uses the **UCI Human Activity Recognition (HAR)** dataset and the **EEG Eye State** dataset. 

The EEG dataset is handled automatically via the `ucimlrepo` Python package. However, the HAR dataset must be downloaded manually due to its specific folder structure requirements.

**To run the HAR experiments:**
1. Download the dataset zip file directly from the UCI repository: 
   [UCI HAR Dataset.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)
2. Extract the zip file directly into the root directory of this repository.
3. Ensure your folder structure looks exactly like this before running the script:
```text
condtsc-implementation/
│
├── CondTSC.py
├── README.md
└── UCI HAR Dataset/          <-- Extracted folder
    ├── test/
    │   └── Inertial Signals/
    └── train/
        └── Inertial Signals/
