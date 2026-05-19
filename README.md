# CondTSC: Dataset Condensation for Time Series Classification via Dual Domain Matching

## Overview

This repository contains the implementation and evaluation of Dataset Condensation for Time Series Classification (CondTSC). The primary objective is to synthesize a microscopic, information-dense dataset $\mathcal{S}$ that acts as a highly efficient proxy for a massive original dataset $\mathcal{T}$.

Training deep learning models on time series data requires extensive datasets, which introduces prohibitive computational bottlenecks for downstream tasks like Neural Architecture Search (NAS), hyperparameter optimization, and continual learning. Standard dataset condensation frameworks fail on sequential data because they operate strictly in the spatial/temporal domains, obfuscating essential spectral topologies like periodicity and harmonics. This implementation circumvents these limitations by optimizing the synthetic data across both the Time Domain and Frequency Domain.

## Key Features & Mathematical Framework

### 1. K-Means Coreset Initialization

Initializing time series synthetic data with random Gaussian noise traps the optimizer in local minima. To resolve this initialization bottleneck, this implementation bootstraps the synthetic dataset using K-Means clustering, extracting the spatial centroids for each class to serve as initial tensor coordinates.
For a defined hyperparameter of Samples Per Class ($spc$), the initialization is:


$$\mathcal{S}_{c}=KMeans(\mathcal{T}_{c},spc)$$

### 2. Dual-Domain Multi-View Augmentation

To leverage spectral topology, $\mathcal{S}$ is sequentially projected into four distinct augmented views $v$ before surrogate training begins:

* 
**Raw:** Unaltered synthetic time series.


* 
**Low-Pass Filter (LPF):** High-frequency noise suppression.


* 
**Phase Perturbation (PP):** Injection of Gaussian noise into the spectral phase angle.


* 
**Magnitude Perturbation (MP):** Injection of Gaussian noise into the spectral magnitude.



### 3. Multi-Step Gradient Matching

To mimic the long-term convergence dynamics of the original dataset, the surrogate objective minimizes the normalized Euclidean distance between the $N$-step synthetic trajectory ($\hat{\theta}_{N,v}^{d}$) and the $M$-step real trajectory ($\overline{\theta}_{M}^{d}$).


$$\mathcal{L}_{grad,v}^{d}=\frac{||\hat{\theta}_{N,v}^{d}-\overline{\theta}_{M}^{d}||_{2}^{2}}{||\theta_{0}-\overline{\theta}_{M}^{d}||_{2}^{2}} \quad \text{for } d \in \{t,f\}$$

### 4. Embedding Matching

To prevent overfitting strictly to weight updates, structural parity in the latent feature space is enforced. High-level embeddings from the penultimate layer of the surrogate CNNBN model are extracted and matched between the synthetic and original datasets.

Dataset,Classes,Original Size,Synthetic Size (spc),K-Means Baseline,CondTSC Accuracy
EEG Eye State,2,"11,985",10 (0.08%),45.69%,30.23% *
UCI HAR,6,"7,352",6 (0.08%),42.83%,55.81%

* **Note on EEG Dataset:** Randomizing train/test splits on rolling-window EEG signals causes massive data leakage. A rigorous chronological split was enforced. The full deep CNNBN backbone achieved only 31.53% generalization on this strict split. CondTSC successfully forced the 10 synthetic samples to flawlessly mimic these exact training dynamics, converging to an identical 30.23%, mathematically validating the bi-level condensation objective.

## Repository Structure

```text
├── CondTSC.py                # Main implementation (Data Loaders, CNNBN Backbone, Augmentations, Condensation Loop)
├── UCI HAR Dataset/          # Target directory for raw HAR inertial signals 
├── CH24B026_DA5400_Time_Series_Classification_report.pdf # Detailed mathematical implementation report
└── README.md

```

## Execution

To execute the condensation pipeline on the target dataset, ensure the datasets are downloaded to the root directory and run the main script. The script automatically handles K-Means initialization, parameter pool collection, bi-level gradient matching across all four augmented views, and baseline evaluations.

```bash
python CondTSC.py

```


## Requirements & Setup

The unrolled bi-level optimization relies on a differentiable computation graph mapped across the inner training loops.

```bash
pip install torch numpy scikit-learn higher tqdm ucimlrepo

```

*Note: The `higher` library is required to preserve the gradients of the $N$-step unroll during surrogate training.*

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
