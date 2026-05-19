"""
CondTSC: Dataset Condensation for Time Series Classification
via Dual Domain Matching — Full Implementation
"""

import os
import time
import copy
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import higher

from tqdm import tqdm


warnings.filterwarnings('ignore')

# Auto-detect GPU for massive speedup in condensation loops
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True # Boosts static-shape CNN speed

torch.manual_seed(42)
np.random.seed(42)

# ════════════════════════════════════════════════════════════════
# 0. DATA LOADER 
# ════════════════════════════════════════════════════════════════

def load_real_har_dataset():
    """Loads raw 9-channel Inertial Signals from UCI HAR, standardizes, and returns tensors."""
    print("Loading raw 9-channel Inertial Signals from UCI HAR Dataset...")

    # The 9 raw signal filenames provided by the UCI dataset
    filenames = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]

    def load_signal_group(group_name):
        filepath = f"UCI HAR Dataset/{group_name}/Inertial Signals/"
        signals = []
        for name in filenames:
            filename = f"{filepath}{name}_{group_name}.txt"
            signals.append(np.loadtxt(filename))
        
        # np.dstack stacks the 9 arrays of shape (N, 128) into (N, 128, 9)
        # .transpose(0, 2, 1) rearranges it to PyTorch's required (Batch, Channels, Length) -> (N, 9, 128)
        return np.dstack(signals).transpose(0, 2, 1)

    # 1. Load the raw 3D data
    X_train = load_signal_group('train')
    X_test  = load_signal_group('test')

    # Load labels (and convert to 0-indexed)
    y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt") - 1
    y_test  = np.loadtxt("UCI HAR Dataset/test/y_test.txt") - 1

    # 2. Standardize the data
    # To scale a 3D array, we must temporarily flatten the spatial dimension
    N_tr, C, L = X_train.shape
    N_te = X_test.shape[0]

    X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, C)  # Shape: (N * 128, 9)
    X_test_flat  = X_test.transpose(0, 2, 1).reshape(-1, C)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_tr, L, C).transpose(0, 2, 1)
    X_test_scaled  = scaler.transform(X_test_flat).reshape(N_te, L, C).transpose(0, 2, 1)

    # 3. Convert to PyTorch tensors
    return (
        torch.FloatTensor(X_train_scaled), 
        torch.LongTensor(y_train.astype(int)),
        torch.FloatTensor(X_test_scaled),  
        torch.LongTensor(y_test.astype(int))
    )
def load_eeg_dataset(window_size=64, test_size=0.2):
    """
    Fetches the EEG Eye State dataset (ID: 264), segments the continuous signal 
    into rolling windows, standardizes, and returns PyTorch tensors.
    """
    print("Loading EEG Eye State Dataset via ucimlrepo...")
    try:
        from ucimlrepo import fetch_ucirepo
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("Please install ucimlrepo: pip install ucimlrepo")

    # 1. Fetch dataset
    eeg_eye_state = fetch_ucirepo(id=264)
    
    # Extract features and targets as numpy arrays
    X_raw = eeg_eye_state.data.features.values
    y_raw = eeg_eye_state.data.targets.values.flatten()
    
    # 2. Segment into continuous sliding windows (Stride = 1)
    # This transforms the (14980, 14) array into (N, window_size, 14)
    X_windows = []
    y_windows = []
    
    for i in range(len(X_raw) - window_size + 1):
        X_windows.append(X_raw[i : i + window_size])
        # Assign the label based on the majority state within the time window
        window_labels = y_raw[i : i + window_size]
        y_windows.append(np.bincount(window_labels).argmax())
        
    X_windows = np.array(X_windows)  # Shape: (14917, 64, 14)
    y_windows = np.array(y_windows)  # Shape: (14917,)
    
    # 3. Train/Test Split (80/20 to match the paper's ~11985/2995 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows, y_windows, test_size=test_size, random_state=42, shuffle=False
    )
    
    # 4. Standardize the data
    N_tr, L, C = X_train.shape
    N_te = X_test.shape[0]
    
    # Flatten spatial dimension for StandardScaler: (N * L, C)
    X_train_flat = X_train.reshape(-1, C)
    X_test_flat  = X_test.reshape(-1, C)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_tr, L, C)
    X_test_scaled  = scaler.transform(X_test_flat).reshape(N_te, L, C)
    
    # 5. Convert to PyTorch Tensors and permute to (Batch, Channels, Length)
    # PyTorch 1D Conv layers strictly require the channel dimension to be index 1.
    X_train_t = torch.FloatTensor(X_train_scaled).permute(0, 2, 1)
    X_test_t  = torch.FloatTensor(X_test_scaled).permute(0, 2, 1)
    
    y_train_t = torch.LongTensor(y_train)
    y_test_t  = torch.LongTensor(y_test)
    
    return X_train_t, y_train_t, X_test_t, y_test_t
# ════════════════════════════════════════════════════════════════
# 1. BACKBONE: 3-layer CNN with BatchNorm (CNNBN)
# ════════════════════════════════════════════════════════════════

class CNNBN(nn.Module):
    def __init__(self, in_channels, num_classes, hidden=32): # as per the paper the hidden was 128
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.MaxPool1d(2),
            
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.MaxPool1d(2),
            
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1), 
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, return_embedding=False): # embeddings in this case refers activation of the last layer convolution layer, right before its fed to softmax
        emb = self.features(x).squeeze(-1) # we want our embeddings of our target dataset and synthetic data set to be similar
        if return_embedding:
            return emb
        return self.classifier(emb)


# ════════════════════════════════════════════════════════════════
# 2. AUGMENTATION MODULE (Frequency Domain via FFT)
# ════════════════════════════════════════════════════════════════

def low_pass_filter(x, keep_ratio=0.5): # we use fast fourier transform to go to frequency domain. of all the frequencies in our data, we remove the higher half and we keep the lower frequency
    X_f = torch.fft.rfft(x, dim=-1)  # this gives us magnitude and the frequncies
    cutoff = max(1, int(X_f.shape[-1] * keep_ratio)) 
    mask = torch.zeros_like(X_f)
    mask[..., :cutoff] = 1.0
    return torch.fft.irfft(X_f * mask, n=x.shape[-1])

def phase_perturbation(x, noise_std=0.1): # we purposefully add noise to our data to improve generalizability
    X_f = torch.fft.rfft(x, dim=-1) # here we add noise to the frequencies
    mag = X_f.abs()
    phase = X_f.angle() + torch.randn_like(X_f.angle()) * noise_std
    return torch.fft.irfft(torch.polar(mag, phase), n=x.shape[-1])

def magnitude_perturbation(x, noise_std=0.1):
    X_f = torch.fft.rfft(x, dim=-1) # here we add noise to the magnitude
    mag = torch.clamp(X_f.abs() + torch.randn_like(X_f.abs()) * noise_std, min=0) # clamping is done here to prevent magnitude from being negative
    return torch.fft.irfft(torch.polar(mag, X_f.angle()), n=x.shape[-1])


def to_freq_domain(x):
    return torch.fft.rfft(x, dim=-1).abs() # inverse fourier transform to get back to time domain


# ════════════════════════════════════════════════════════════════
# 3. K-MEANS INITIALISATION
# ════════════════════════════════════════════════════════════════
# the paper mentioned that we cannot randomly initialize the values in the synthetic data set
# hence we use k means clustering for each class, with k value equal to the number of synthetic samples per class (spc)
# the centroids of each cluster is an observation of the synthetic dataset

def kmeans_init(X_train, y_train, num_classes, spc): 
    S_list, lbl_list = [], [] # list of observations and list of its labels, respectively
    X_np = X_train.numpy().reshape(len(X_train), -1) 
    for c in range(num_classes):
        idx = np.where(y_train.numpy() == c)[0]
        km = KMeans(n_clusters=spc, random_state=42, n_init=10)
        km.fit(X_np[idx])
        C, L = X_train.shape[1], X_train.shape[2]
        centroids = torch.FloatTensor(km.cluster_centers_.reshape(spc, C, L))
        S_list.append(centroids)
        lbl_list.append(torch.full((spc,), c, dtype=torch.long))
    return torch.cat(S_list).to(DEVICE), torch.cat(lbl_list).to(DEVICE) # returns the list of observations and labels


# ════════════════════════════════════════════════════════════════
# 4. TRAINING HELPERS
# ════════════════════════════════════════════════════════════════


def collect_param_pool(X, y, in_ch, num_classes, n_models=20, epochs=3): # we draw the initial parameters of our main model all at once at random and then we use it for each "epoch"
    """Build p(theta) by collecting checkpoints from short training runs."""
    pool = []
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    
    for _ in tqdm(range(n_models), desc="Building Parameter Pool", leave=False):
        m = CNNBN(in_ch, num_classes).to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        pool.append(copy.deepcopy(m.state_dict()))
    return pool


# ════════════════════════════════════════════════════════════════
# 5. DUAL SURROGATE LOSSES
# ════════════════════════════════════════════════════════════════


# the paper has used multistep gradient matching as a surrogate objective, 
# L2 norm of difference between the final parameters of the main models 
# built on the original dataset and the synthetic dataset
# is divided by the L2 norm of the difference between the initial parameter (common for both models) and the original model

# notice that only p1 is a function of the synthetic dataset
def grad_match_loss(params_hat, params_bar, params_0):
    num = sum(((p1 - p2) ** 2).sum() for p1, p2 in zip(params_hat, params_bar))
    den = sum(((p0 - p2) ** 2).sum() for p0, p2 in zip(params_0, params_bar)) + 1e-8 # added for numerical stability to prevent zero denominator
    return num / den

# this is the embedding matching and the second surrogate objective of the paper
def emb_match_loss(model_hat, model_bar, S_domain):
    model_hat.eval(); model_bar.eval()
    H_hat = model_hat(S_domain, return_embedding=True)
    H_bar = model_bar(S_domain, return_embedding=True)
    return ((H_hat.mean(0) - H_bar.mean(0)) ** 2).sum()


# ════════════════════════════════════════════════════════════════
# 6. MAIN CondTSC TRAINING LOOP
# ════════════════════════════════════════════════════════════════

import higher
# paper had 2000 epochs and learning rate as 1

def condtsc_train(X_train, y_train, num_classes, spc,
                  n_epochs=2000, N=10, M=1000,
                  lam=1.0, lr_S=1.0, param_pool=None):
    
    C, L = X_train.shape[1], X_train.shape[2]
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    crit = nn.CrossEntropyLoss()

    S, S_labels = kmeans_init(X_train.cpu(), y_train.cpu(), num_classes, spc)
    S = S.clone().detach().requires_grad_(True)
    opt_S = optim.SGD([S], lr=lr_S) 

    if param_pool is None:
        print("  Building parameter pool...")
        param_pool = collect_param_pool(X_train, y_train, C, num_classes, n_models=15, epochs=2)

    pbar = tqdm(range(n_epochs), desc=f"Condensing spc={spc} (Multi-Step)")
    for epoch in pbar:
        opt_S.zero_grad()
        
        # S retains its gradients through multi_view_augment

        epoch_loss_display = 0.0 
        vnames = ['raw', 'LPF', 'PP', 'MP']

        for vname in vnames:
            if vname == 'raw':
                S_aug = S
            elif vname == 'LPF':
                S_aug = low_pass_filter(S)
            elif vname == 'PP':
                S_aug = phase_perturbation(low_pass_filter(S))
            elif vname == 'MP':
                S_aug = magnitude_perturbation(phase_perturbation(low_pass_filter(S))) 
                S_t = S_aug
                S_f = to_freq_domain(S_aug)

                T_idx = torch.randperm(len(X_train))[:min(512, len(X_train))]
                T_t = X_train[T_idx]
                T_f = to_freq_domain(T_t)
                T_y = y_train[T_idx]

                sd0 = random.choice(param_pool)

                # ==========================================================
                # 1. TIME DOMAIN MULTI-STEP MATCHING
                # ==========================================================
                m_t_syn = CNNBN(C, num_classes).to(DEVICE)
                m_t_syn.load_state_dict(sd0)
                opt_t_syn = optim.SGD(m_t_syn.parameters(), lr=0.001)
                
                # [PAPER FIX 1]: Capture the true initial parameters for the denominator
                theta_0_t = [p.clone().detach() for p in m_t_syn.parameters()]

                m_t_real = CNNBN(C, num_classes).to(DEVICE)
                m_t_real.load_state_dict(sd0)
                opt_t_real = optim.SGD(m_t_real.parameters(), lr=0.001)

                # --- Real Data Trajectory (M steps) ---
                m_t_real.train()
                for _ in range(M):
                    batch_idx = torch.randperm(len(T_t))[:128]
                    loss_real = crit(m_t_real(T_t[batch_idx]), T_y[batch_idx])
                    opt_t_real.zero_grad()
                    loss_real.backward()
                    opt_t_real.step()
                
                theta_bar_M_t = [p.detach() for p in m_t_real.parameters()]

                # --- Synthetic Data Trajectory (N steps) ---
                with higher.innerloop_ctx(m_t_syn, opt_t_syn, copy_initial_weights=False) as (fmodel_t, diffopt_t):
                    for _ in range(N):
                        loss_syn = crit(fmodel_t(S_t), S_labels)
                        diffopt_t.step(loss_syn)
                    
                    theta_hat_N_t = list(fmodel_t.parameters())
                    
                    Le_t = ((fmodel_t(S_t, return_embedding=True).mean(0) - m_t_real(T_t, return_embedding=True).detach().mean(0))**2).sum()

                Lg_t = grad_match_loss(theta_hat_N_t, theta_bar_M_t, theta_0_t)

                # ==========================================================
                # 2. FREQUENCY DOMAIN MULTI-STEP MATCHING
                # ==========================================================
                m_f_syn = CNNBN(C, num_classes).to(DEVICE)
                m_f_syn.load_state_dict(sd0)
                opt_f_syn = optim.SGD(m_f_syn.parameters(), lr=0.01)
                
                # [PAPER FIX 2]: Capture the true initial parameters for the denominator
                theta_0_f = [p.clone().detach() for p in m_f_syn.parameters()]

                m_f_real = CNNBN(C, num_classes).to(DEVICE)
                m_f_real.load_state_dict(sd0)
                opt_f_real = optim.SGD(m_f_real.parameters(), lr=0.01)

                # --- Real Data Trajectory (M steps) ---
                m_f_real.train()
                for _ in range(M):
                    batch_idx = torch.randperm(len(T_f))[:128]
                    loss_real = crit(m_f_real(T_f[batch_idx]), T_y[batch_idx])
                    opt_f_real.zero_grad()
                    loss_real.backward()
                    opt_f_real.step()
                
                theta_bar_M_f = [p.detach() for p in m_f_real.parameters()]

                # --- Synthetic Data Trajectory (N steps) ---
                with higher.innerloop_ctx(m_f_syn, opt_f_syn, copy_initial_weights=False) as (fmodel_f, diffopt_f):
                    for _ in range(N):
                        loss_syn = crit(fmodel_f(S_f), S_labels)
                        diffopt_f.step(loss_syn)
                    
                    theta_hat_N_f = list(fmodel_f.parameters())
                    
                    Le_f = ((fmodel_f(S_f, return_embedding=True).mean(0) - m_f_real(T_f, return_embedding=True).detach().mean(0))**2).sum()

                Lg_f = grad_match_loss(theta_hat_N_f, theta_bar_M_f, theta_0_f)

                # ==========================================================
                # 3. GRAPH FLUSH & ACCUMULATION
                # ==========================================================
                # Calculate total loss for THIS VIEW ONLY
                view_loss = Lg_t + Lg_f + lam * (Le_t + Le_f)
                
                # [MEMORY FIX]: Backward pass inside the loop! 
                # This pushes gradients directly to S and instantly frees the massive higher graph from VRAM.
                view_loss.backward()
                
                epoch_loss_display += view_loss.item()

            # Step the optimizer once per epoch after all 4 views have accumulated gradients onto S
            opt_S.step()

            pbar.set_postfix({'loss': f"{(epoch_loss_display / 4):.4f}"})

    return S.detach(), S_labels

# ════════════════════════════════════════════════════════════════
# 7. EVALUATION
# ════════════════════════════════════════════════════════════════

def evaluate_condensed(S, S_labels, X_test, y_test, in_ch, num_classes, n_runs=5, train_epochs=300):
    accs = []
    crit = nn.CrossEntropyLoss()
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    S = S.to(DEVICE); S_labels = S_labels.to(DEVICE)

    for run in range(n_runs):
        torch.manual_seed(run)
        model = CNNBN(in_ch, num_classes).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        ds    = TensorDataset(S, S_labels)
        loader= DataLoader(ds, batch_size=len(S), shuffle=True)

        model.train()
        # <-- tqdm added here
        for ep in tqdm(range(train_epochs), desc=f"Eval Condensed Run {run+1}/{n_runs}", leave=False):
            for xb, yb in loader:
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_test).argmax(1)
            acc  = (pred == y_test).float().mean().item() * 100
        accs.append(acc)

    return np.mean(accs), np.std(accs)

def evaluate_full(X_train, y_train, X_test, y_test, in_ch, num_classes, train_epochs=300):
    model = CNNBN(in_ch, num_classes).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    ds    = TensorDataset(X_train.to(DEVICE), y_train.to(DEVICE))
    loader= DataLoader(ds, batch_size=256, shuffle=True)
    
    model.train()
    # <-- tqdm added here
    for ep in tqdm(range(train_epochs), desc="Evaluating Full Dataset", leave=False):
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(DEVICE)).argmax(1).cpu()
    return accuracy_score(y_test.cpu().numpy(), pred.numpy()) * 100

def run_classical_baselines(X_tr, y_tr, X_te, y_te):
    X_tr_flat = X_tr.cpu().numpy().reshape(len(X_tr), -1)
    X_te_flat = X_te.cpu().numpy().reshape(len(X_te), -1)
    y_tr_np = y_tr.cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_flat)
    X_te_s = scaler.transform(X_te_flat)

    results = {}
    for name, clf in [
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('SVM (RBF)',     SVC(kernel='rbf', C=1.0, random_state=42)),
        ('KNN (k=5)',     KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]:
        t0 = time.time()
        clf.fit(X_tr_s, y_tr_np)
        pred = clf.predict(X_te_s)
        acc = accuracy_score(y_te_np, pred) * 100
        elapsed = time.time() - t0
        results[name] = acc
        print(f"  {name:<20s}: {acc:.2f}%  ({elapsed:.1f}s)")
    return results


# ════════════════════════════════════════════════════════════════
# 8. RUN EXPERIMENTS
# ════════════════════════════════════════════════════════════════

def run_dataset(name, X_tr, y_tr, X_te, y_te, num_classes, spc_list=[1, 5, 10], n_epochs=2000):
    C = X_tr.shape[1]
    print(f"\n{'='*65}")
    print(f"DATASET: {name}  |  train={len(X_tr)}  test={len(X_te)}"
          f"  C={C}  L={X_tr.shape[2]}  K={num_classes}")
    print(f"Device: {DEVICE.upper()}")
    print('='*65)

    print("\n--- Classical Baselines (full training data) ---")
    baseline_results = run_classical_baselines(X_tr, y_tr, X_te, y_te)

    print("\n--- Full CNNBN ---")
    full_acc = evaluate_full(X_tr, y_tr, X_te, y_te, C, num_classes)
    print(f"  CNNBN (full): {full_acc:.2f}%")

    print("\n--- Building parameter pool ---")
    pool = collect_param_pool(X_tr.to(DEVICE), y_tr.to(DEVICE), C, num_classes, n_models=10, epochs=2)

    cond_results = {}
    for spc in spc_list:
        ratio = spc * num_classes / len(X_tr) * 100
        print(f"\n--- CondTSC  spc={spc}  ratio={ratio:.2f}% ---")
        t0 = time.time()
        S, S_lbl = condtsc_train(X_tr, y_tr, num_classes, spc,
                                  n_epochs=n_epochs, N=10, M=1000,
                                  lam=1.0, lr_S=1, param_pool=pool)
        elapsed = time.time() - t0
        mu, std = evaluate_condensed(S, S_lbl, X_te, y_te, C, num_classes, n_runs=3, train_epochs=300)
        cond_results[spc] = (mu, std, ratio)
        print(f"  CondTSC spc={spc}: {mu:.2f}±{std:.2f}%  ({elapsed:.0f}s condensation)")

    kmeans_results = {}
    print("\n--- Evaluating K-Means Coreset ---")
    for spc in spc_list:
        S_km, lbl_km = kmeans_init(X_tr, y_tr, num_classes, spc)
        mu_km, std_km = evaluate_condensed(S_km, lbl_km, X_te, y_te, C, num_classes, n_runs=3, train_epochs=300)
        kmeans_results[spc] = (mu_km, std_km)

    print(f"\n{'─'*65}")
    print(f"{'Method':<22} {'spc='+str(spc_list[0]):<18} "
          f"{'spc='+str(spc_list[1] if len(spc_list) > 1 else '-'):<18} "
          f"{'spc='+str(spc_list[2] if len(spc_list) > 2 else '-'):<18}")
    print(f"{'─'*65}")

    km_row = "K-means (coreset)"
    print(f"{km_row:<22}", end="")
    for spc in spc_list:
        mu, std = kmeans_results[spc]
        print(f"  {mu:.2f}±{std:.2f}%    ", end="")
    print()

    c_row = "CondTSC"
    print(f"{c_row:<22}", end="")
    for spc in spc_list:
        mu, std, ratio = cond_results[spc]
        print(f"  {mu:.2f}±{std:.2f}%    ", end="")
    print()

    print(f"{'─'*65}")
    for bname, bacc in baseline_results.items():
        print(f"{bname:<22}  {bacc:.2f}% (full data)")
    print(f"{'CNNBN (full)':<22}  {full_acc:.2f}% (full data)")
    print(f"{'─'*65}")

    return {
        'baselines': baseline_results,
        'full_cnnbn': full_acc,
        'condtsc': cond_results,
        'kmeans': kmeans_results,
    }


# if __name__ == '__main__':
#     try:
#         Xtr_h, ytr_h, Xte_h, yte_h = load_real_har_dataset()
#     except FileNotFoundError as e:
#         print("\n[!] Error: Could not find the 'UCI HAR Dataset' folder.")
#         print("[!] Please execute this script from the directory containing the 'UCI HAR Dataset' folder.")
#         exit(1)
#     har_results = run_dataset(
#         name='UCI HAR (Real Data)', 
#         X_tr=Xtr_h, 
#         y_tr=ytr_h, 
#         X_te=Xte_h, 
#         y_te=yte_h,
#         num_classes=6, 
#         spc_list=[1],  
#         n_epochs=200          
#     )

if __name__ == '__main__':
    Xtr, ytr, Xte, yte = load_eeg_dataset(window_size=64)

    eeg_results = run_dataset(
        name='EEG Eye State (Real Data)', 
        X_tr=Xtr, 
        y_tr=ytr, 
        X_te=Xte, 
        y_te=yte,
        num_classes=2, # EEG is a binary classification task
        spc_list=[1], 
        n_epochs=200
    )
    print("\n\nEXPERIMENT COMPLETE.")