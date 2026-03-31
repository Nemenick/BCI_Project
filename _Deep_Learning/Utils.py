import numpy as np
import scipy.signal as sc_sig
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error)

############################# Simple functions #############################

def freq_filter(signal,sf,freqs,type_filter="bandpass", order_filter=4, axis=-1):
    """ 
    sf: sampling rate of the input waveform
    freqs: list of frequences (e.g. 2 for bandpass), or single float (e.g. for highpass)
        sf sampling frequence  """
    # type_filter: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’

    freqs=np.array(freqs)
    filt_b1,filt_a1=sc_sig.butter(order_filter,freqs/(sf/2),btype=type_filter)
    filtered_sig=sc_sig.filtfilt(filt_b1,filt_a1,sc_sig.detrend(signal, axis=axis), axis=axis) # both detrend and filtfilt have axis = -1 by default OK!
    return filtered_sig


def extract_windows(x, win_len=128, shift=62):
    """ OK, verified (see Test.ipynb -> Function extract window check)"""
    # x shape: (192, 68, 748) 
    # 192 trials, 68 ROIs, 748 time points
    # 128 + 62*(11-1)  = 748

    windows = sliding_window_view(x, window_shape=win_len, axis=-1)
    # windows shape: (192, 68, 748 - win_len + 1, win_len)

    # apply hop/shift
    windows = windows[..., ::shift, :]

    return windows


def expand_trials_to_regions(X, y):
    """
    X: (n_trials, n_regions, n_timepoints)
    y: (n_trials,) ∈ {-1, +1}

    Returns:
        X_out: (n_trials * n_regions, n_timepoints)
        y_out: (n_trials * n_regions,) ∈ {-R,...,-1,+1,...,+R}
    """

    X = np.asarray(X)
    y = np.asarray(y)

    n_trials, n_regions, n_timepoints = X.shape

    region_ids = np.arange(1, n_regions + 1)
    region_ids = np.broadcast_to(region_ids, (n_trials, n_regions))
    # region_ids shape: (n_trials, n_regions) [[1,2,3,...,R], [1,2,3,...,R], ...]
    

    trial_signs = y[:, None]
    trial_signs = np.broadcast_to(trial_signs, (n_trials, n_regions))
    # trial_signs shape: (n_trials, n_regions) [[-1,-1,-1,...,-1], [-1,-1,-1,...,-1], ... , [1,1,1,...,1]]

    # y1 shape: (n_trials, n_regions);
    # valore abs in base a ROI (column), da 1 a 68
    # segno in base a Rest/MI (riga), - per Rest, + per MI
    y1 = trial_signs * region_ids

    # faccio reshape "conforme" tra i due array
    X_out = X.reshape(n_trials * n_regions, n_timepoints)
    y_out = y1.reshape(n_trials * n_regions)

    return X_out, y_out


############################ "Composite" functions #############################


def five_folds_train_val_test(
    X, y,
    val_fraction_of_block=0.5,   # 0.5 → 10% val + 10% test
    seed=224
):
    """
    Function that splits in training, validation and test sets using 5-Fold StratifiedKFold.
    Coherent split with training of classical methods (see ../Codes/Trainings.ipynb)

    X: (n_trials, n_regions, n_timepoints)      order is Rest, then MI
    y: (n_trials,) ∈ {-1, +1}                   order is Rest, then MI

    Outputs:
    X_train: (n_trials_train, n_regions, n_timepoints)      order is Rest, then MI
    y_train: (n_trials_train,) ∈ {-1, +1}                   order is Rest, then MI
    . . . (same for val and test)

    """

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, block_idx) in enumerate(skf.split(X, y)):
        
        # sto inserendo nel training tutte le regioni relative a "quel trial"
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_block = X[block_idx]
        y_block = y[block_idx]

        # Split block into val / test
        X_val, X_test, y_val, y_test = train_test_split(
            X_block,
            y_block,
            test_size=1 - val_fraction_of_block,
            stratify=y_block,
            random_state=seed
        )

        yield (
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
            train_idx
        )


def windowize(X,Y, win_len=128, shift=62):
    """
    Function to extract windows from X and replicate labels Y accordingly 
    to assign the same label to all windows extracted from a single trace (trial)
    
    X: (n_traces, n_timepoints)
    Y: (n_traces,) (-68,-67,...,-1,1,2,...,68) sign based on Rest/MI, absolute value is region index
    win_len: timepoints of extracted windows
    shift: shift between windows (if shift < win_len, windows will overlap, desiderable for data augmentation)
    """

    X_windowed = extract_windows(X, win_len, shift)
    Y_windowed = np.repeat(Y[:, None], X_windowed.shape[1], axis=1)
    # X_windowed.shape, Y_windowed.shape
    # (n_traces, n_windows_per_trace, win_len), (n_traces, n_windows_per_trace)

    X_windowed_reshaped = X_windowed.reshape(-1, X_windowed.shape[-1])
    Y_windowed_reshaped = Y_windowed.reshape(-1,)
    
    # X_windowed_reshaped.shape, Y_windowed_reshaped.shape
    # (n_traces*n_windows_per_trace, win_len), (n_traces*n_windows_per_trace,)

    return X_windowed_reshaped, Y_windowed_reshaped


class RegionWiseStandardizer:
    """
    Standardize time-series traces using per-label mean and std
    computed ONLY on training data.
    """

    def __init__(self, eps=1e-15):
        self.eps = eps
        self.mu_ = {}
        self.std_ = {}
        self.regions = None
        self.fitted_ = False

    def fit(self, X, y):
        """
        X: array (n_samples, n_timepoints)
        y: array (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D (samples, timepoints)")
        if y.ndim != 1:
            raise ValueError("y must be 1D (samples,)")

        self.regions = np.unique(np.abs(y))

        for region in self.regions:
            X_region = X[np.abs(y) == region]

            mu = X_region.mean()
            std = X_region.std()

            #std[std < self.eps] = 1.0
            # print(X_region.shape, mu.shape, std.shape)
            # print(std)

            self.mu_[region] = mu
            self.std_[region] = std

        self.fitted_ = True
        return self

    def transform(self, X, y):
        """
        Normalize X using region-dependent statistics.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform()")

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D (samples, timepoints)")
        if y.ndim != 1:
            raise ValueError("y must be 1D (samples,)")

        Xn = np.empty_like(X, dtype=float)

        for region in np.unique(np.abs(y)):
            if region not in self.mu_:
                raise ValueError(f"Region {region} not seen during fit")

            idx = (np.abs(y) == region)
            Xn[idx] = (X[idx] - self.mu_[region]) / self.std_[region]

        return Xn

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)


class TraceWiseStandardizer:

    def __init__(self, method="zscore"):
        self.method = method

    def zscore_normalizer(self, X):
        mu = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (X - mu) / std

    def minmax_normalizer(self, X):
        minv = X.min(axis=1, keepdims=True)
        maxv = X.max(axis=1, keepdims=True)
        denom = maxv - minv
        denom[denom == 0] = 1.0
        return (X - minv) / denom

    def transform(self, X):
        if self.method == "zscore":
            return self.zscore_normalizer(X)
        elif self.method == "minmax":
            return self.minmax_normalizer(X)
        else:
            raise ValueError("Unknown method")


############################## Saving results / Performances #############################

def save_training_results(model, history, save_dir):
    """
    Save:
    - Keras model
    - training history as CSV
    - one plot per metric (train + val)
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------
    # 1. Save weights
    # ---------------------------
    weights_path = os.path.join(save_dir, "model.weights.h5")
    model.save_weights(weights_path)
    print(f"Model weights saved to {weights_path}")

    # ---------------------------
    # 2. History -> DataFrame
    # ---------------------------
    hist_df = pd.DataFrame(history.history)
    csv_path = os.path.join(save_dir, "history.csv")
    hist_df.to_csv(csv_path, index=False)
    print(f"History saved to {csv_path}")

    # ---------------------------
    # 3. Plot metrics
    # ---------------------------
    for key in hist_df.columns:

        # skip validation keys here (handled together)
        if key.startswith("val_"):
            continue

        plt.figure()
        plt.plot(hist_df[key], label=f"train_{key}")

        val_key = f"val_{key}"
        if val_key in hist_df.columns:
            plt.plot(hist_df[val_key], label=val_key)

        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(key)
        plt.legend()
        plt.grid(True)

        fig_path = os.path.join(save_dir, f"{key}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Plots saved to {save_dir}")

def evaluate_multitask_by_region(model,x_test,y_test,x_test_output=None,threshold=0.5,save_name=None):
    """
    Computes reconstruction and/or classification metrics grouped by region.
    Works with:
        - autoencoder
        - classifier
        - supervised autoencoder
    """

    outputs = model.predict(x_test)

    # -----------------------------------
    # Detect model outputs
    # -----------------------------------

    x_test_output_shape = x_test.shape if x_test_output is None else x_test_output.shape

    
    if isinstance(outputs, dict):
        x_recon = outputs.get("reconstruction", None)
        y_prob = outputs.get("classification", None)
    else:
        # single-head models
        print(outputs.shape, x_test_output_shape)
        if outputs.shape == x_test_output_shape:  # reconstruction
            x_recon = outputs
            y_prob = None
        else:                 # classification
            x_recon = None
            y_prob = outputs.squeeze()

    if y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)

    if x_recon is not None:
        if x_test_output is None:
            x_test_output = x_test

        if x_test_output.shape != x_test.shape:
            print("WARNING: x_test_output.shape != x_test.shape")

    results = []

    for region in np.unique(np.abs(y_test)):
        mask = np.abs(y_test) == region
        n_windows = mask.sum()

        if n_windows == 0:
            continue

        row = {
            "ROIs": region - 1,
            "NROIs": 1,
            "n_windows": n_windows
        }

        # -----------------------------------
        # Classification metrics
        # -----------------------------------
        if y_prob is not None:

            y_true = (y_test[mask] > 0).astype(int)
            y_region_pred = y_pred[mask]

            row.update({
                "Accuracy": accuracy_score(y_true, y_region_pred),
                "Precision": precision_score(y_true, y_region_pred, zero_division=0),
                "Recall": recall_score(y_true, y_region_pred, zero_division=0),
                "F1": f1_score(y_true, y_region_pred, zero_division=0)
            })

        # -----------------------------------
        # Reconstruction metrics
        # -----------------------------------
        if x_recon is not None:

            x_true = x_test_output[mask]
            x_pred = x_recon[mask]

            x_true_f = x_true.reshape(len(x_true), -1)
            x_pred_f = x_pred.reshape(len(x_pred), -1)

            mse = np.mean((x_true_f - x_pred_f) ** 2, axis=1)
            mae = np.mean(np.abs(x_true_f - x_pred_f), axis=1)

            row.update({
                "MSE_mean": mse.mean(),
                "MSE_std": mse.std(),
                "MAE_mean": mae.mean(),
                "MAE_std": mae.std()
            })

        results.append(row)

    df = pd.DataFrame(results).sort_values("ROIs").reset_index(drop=True)

    if save_name is not None:
        df.to_csv(save_name, index=False)
        print(f"Region-wise performance saved to {save_name}")

    return df

def evaluate_autoencoder_by_region(model,x_test,y_test,x_test_output=None,save_name=None):
    """
    Preserving the original autoencoder evaluation API.
    Returns only reconstruction metrics.
    """

    df = evaluate_multitask_by_region(model=model, x_test=x_test, y_test=y_test, x_test_output=x_test_output, save_name=save_name)

    recon_cols = ["ROIs","NROIs","n_windows","MSE_mean","MSE_std","MAE_mean","MAE_std"]

    return df[recon_cols]

def evaluate_classification_by_region( model, x_test, y_test, threshold=0.5, save_name=None):
    """
    Preserving the original classification evaluation API.
    Returns only classification metrics.
    """

    df = evaluate_multitask_by_region(model=model, x_test=x_test, y_test=y_test, threshold=threshold, save_name=save_name)

    class_cols = ["ROIs","NROIs","n_windows","Accuracy","Precision","Recall","F1"]

    return df[class_cols]

def evaluate_encoder_decoder_spectra_by_region(model, x_test, psd_test, y_test, save_name=None):
    """
    Compute reconstruction performance grouped by abs(y_test),
    supporting both EEG-only and EEG+MEG.
    """

    spectra_recon = model.predict(x_test)

    # unwrap dict outputs
    if isinstance(spectra_recon, dict):
        spectra_recon = spectra_recon["reconstruction"]

    spectra_recon = np.asarray(spectra_recon)
    psd_test = np.asarray(psd_test)

    results = []

    n_channels = 1 if psd_test.ndim == 2 else psd_test.shape[-1]

    for region in np.unique(np.abs(y_test)):

        mask = np.abs(y_test) == region

        true_r = psd_test[mask]
        pred_r = spectra_recon[mask]

        if len(true_r) == 0:
            continue

        row = {
            "ROIs": region - 1,
            "n_windows": len(true_r),
        }

        # --- EEG only case ---
        if n_channels == 1:
            axes = tuple(range(1, true_r.ndim))
            mse = np.mean((true_r - pred_r) ** 2, axis=axes)
            mae = np.mean(np.abs(true_r - pred_r), axis=axes)

            row.update({
                "MSE_mean": mse.mean(),
                "MSE_std": mse.std(),
                "MAE_mean": mae.mean(),
                "MAE_std": mae.std(),
            })

        # --- EEG + MEG case ---
        else:
            for ch, name in enumerate(["EEG", "MEG"]):
                t = true_r[..., ch]
                p = pred_r[..., ch]

                axes = tuple(range(1, t.ndim))

                mse = np.mean((t - p) ** 2, axis=axes)
                mae = np.mean(np.abs(t - p), axis=axes)

                row.update({
                    f"MSE_mean_{name}": mse.mean(),
                    f"MSE_std_{name}": mse.std(),
                    f"MAE_mean_{name}": mae.mean(),
                    f"MAE_std_{name}": mae.std(),
                })

        results.append(row)

    df = pd.DataFrame(results).sort_values("ROIs").reset_index(drop=True)

    if save_name is not None:
        df.to_csv(save_name, index=False)
        print(f"Saved region-wise reconstruction performance to {save_name}")

    return df

def aggregate_columns_dfs(dfs,  cols_to_aggregate=["Accuracy", "Precision", "Recall", "F1", "n_windows"]
):
    """
    dfs: list of DataFrames (one per fold) [df1, df2, ...]
    Each dataframe must have a 'region' column.
    
    Columns listed in cols_to_aggregate are aggregated into tuples (len = n_folds).
    Other columns are assumed identical across dfs and copied as scalars.
    
    Returns: aggregated DataFrame
    """

    # Use region as index for all dfs
    dfs = [df.set_index("ROIs") for df in dfs]

    regions = dfs[0].index
    columns = dfs[0].columns

    aggregated_rows = []

    for region in regions:
        row = {"region": region}

        for col in columns:
            if col in cols_to_aggregate:
                row[col] = tuple(df.loc[region, col] for df in dfs)
            else:
                # take the common value (assumed identical across folds)
                row[col] = dfs[0].loc[region, col]

        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)

def plot_random_reconstructions( model, x_test, x_test_output=None, sf=250, freqs_filt=(4, 20), n_samples=10, save_dir="reconstruction_plots", random_state=None, freqs_filt_axis=-1):
    """
    Plot original, reconstructed, and filtered traces for random samples.

    Parameters
    ----------
    model : keras / tf model Trained autoencoder.
    x_test,x_test_out : np.ndarray  Shape: (n_samples, n_times)
    sf : float
        Sampling frequency.
    """
    if x_test_output is not None and x_test_output.shape != x_test.shape:
        print(("WARNING IN plot_random_reconstructions: x_test_out.shape != x_test.shape "))

    if x_test_output is None:
        x_test_output = x_test

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, ".gitignore"), "w") as f:
        f.write("*")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(x_test), size=n_samples, replace=False)

    # Pre-compute filtered signals
    if freqs_filt is not None:
        x_filt_out = freq_filter(x_test_output, sf=sf, freqs=list(freqs_filt), type_filter="bandpass", axis=freqs_filt_axis)

    lw = 2.3

    for k, idx in enumerate(indices):
        # --- reconstruction (EXACTLY like your snippet)
        x_in = x_test[idx:idx+1]
        x_rec = model(x_in)
        if isinstance(x_rec, dict):
            x_rec = x_rec["reconstruction"]
        x_rec = x_rec.numpy().reshape(x_in.shape)

        for _ in range(x_rec.shape[-1]):
            plt.figure(figsize=(10, 4))
            plt.plot(x_test_output[idx,:,_], linewidth=lw, label="original", color="C0")
            plt.plot(x_rec[0, :, _], linewidth=lw, label="reconstructed", color="C1")
            if freqs_filt is not None and _ == 0:
                plt.plot(x_filt_out[idx, :, _], linewidth=lw, label=f"filtered {freqs_filt[0]}-{freqs_filt[1]} Hz)", color="C2")

            plt.title(f"Sample {idx}")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()

            fname = os.path.join(save_dir, f"reconstruction_{k:02d}_idx{idx}_freq_band_n_{_+1}.png")
            plt.savefig(fname, dpi=150)
            plt.close()

def plot_random_PowerSpectra_reconstructed( model, x_test, psd_test, freq_bins, n_samples=10, save_dir="reconstructed_spectra", random_state=224):
    """
    Plot original, reconstructed, and filtered traces for random samples.

    Parameters
    ----------
    model : keras / tf model Trained autoencoder.
    x_test : np.ndarray  Shape: (n_samples, n_times)
    """
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, ".gitignore"), "w") as f:
        f.write("*")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(x_test), size=n_samples, replace=False)

    spectra_rec = model.predict(x_test)
    if isinstance(spectra_rec, dict):
        spectra_rec = spectra_rec["reconstruction"]

    spectra_rec = np.asarray(spectra_rec)
    psd_test = np.asarray(psd_test)

    n_channels = 1 if psd_test.ndim == 2 else psd_test.shape[-1]

    lw = 2.3

    for k, idx in enumerate(indices):

        x_in = x_test[idx:idx+1]
        rec = model(x_in)

        if isinstance(rec, dict):
            rec = rec["reconstruction"]
        rec = np.asarray(rec)[0]

        # -------- EEG only --------
        if n_channels == 1:
            plt.figure(figsize=(10, 4))
            plt.plot(freq_bins, psd_test[idx], label="original", linewidth=lw)
            plt.plot(freq_bins, rec, label="reconstructed", linewidth=lw)

        # -------- EEG + MEG --------
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            labels = ["EEG", "MEG"]

            for ch in range(2):
                axes[ch].plot(freq_bins, psd_test[idx, :, ch], label="original", linewidth=lw)
                axes[ch].plot(freq_bins, rec[:, ch], label="reconstructed", linewidth=lw)
                axes[ch].set_title(labels[ch])
                axes[ch].set_xlabel("Freq")
                axes[ch].legend()

        plt.suptitle(f"Sample {idx}")
        plt.tight_layout()

        fname = os.path.join(save_dir, f"Spectra_reconstructed_{k:02d}_idx{idx}.png")
        plt.savefig(fname, dpi=150)
        plt.close()


################################ Deprecated code (but still in use) ################################

def create_labels(n_trials, n_regions=68):

    # assegno negativi a quelli di prima (Rest) e positivi a quelli di dopo (MI) 
    y = np.ones((n_trials, n_regions))
    y[0:n_trials//2, :] = -1

    # assegno valore in base a ROI, da ±1 a ±68
    for col in range(y.shape[1]):
        y[:, col] *= (col+1)
    return y

def evaluate_classification_by_region_old(model, x_test, y_test, save_name=None, threshold=0.5):
    """
    Compute classification performance grouped by abs(y_test).
    Saves a CSV in save_dir.
    """
    # Model predictions
    y_prob = model.predict(x_test).squeeze()
    y_pred = (y_prob >= threshold).astype(int)

    results = []

    for region in np.unique(np.abs(y_test)):
        mask = np.abs(y_test) == region
        y_true = (y_test[mask] > 0).astype(int)
        y_region_pred = y_pred[mask]

        if len(y_true) == 0:
            continue
        row = {
            "ROIs": region-1,
            "NROIs": len([region]),
            "n_windows": len(y_true),
            "Accuracy": accuracy_score(y_true, y_region_pred),
            "Precision": precision_score(y_true, y_region_pred, zero_division=0),
            "Recall": recall_score(y_true, y_region_pred, zero_division=0),
            "F1": f1_score(y_true, y_region_pred, zero_division=0),
        }

        results.append(row)
    df = pd.DataFrame(results).sort_values("ROIs").reset_index(drop=True)

    # Save
    if save_name is not None:
        df.to_csv(save_name, index=False)

    print(f"Region-wise performance saved to {save_name}")

    return df

def evaluate_autoencoder_by_region_old(model, x_test, y_test, x_test_output=None, save_name=None):
    """
    Compute reconstruction performance grouped by abs(y_test).
    """
    if x_test_output is not None and x_test_output.shape != x_test.shape:
        print(("WARNING IN plot_random_reconstructions: x_test_out.shape != x_test.shape "))

    if x_test_output is None:
        x_test_output = x_test

    # Reconstruct
    x_recon = model.predict(x_test)

    results = []

    for region in np.unique(np.abs(y_test)):
        mask = np.abs(y_test) == region
        x_true_output = x_test_output[mask]
        x_pred = x_recon[mask]

        if len(x_true_output) == 0:
            continue

        # Flatten per-sample for error computation
        x_true_f = x_true_output.reshape(len(x_true_output), -1)
        x_pred_f = x_pred.reshape(len(x_pred), -1)

        # Per-sample errors
        mse_per_sample = np.mean((x_true_f - x_pred_f) ** 2, axis=1)
        mae_per_sample = np.mean(np.abs(x_true_f - x_pred_f), axis=1)

        row = {
            "ROIs": region-1,
            "NROIs": len([region]),
            "n_windows": len(x_true_output),
            "MSE_mean": mse_per_sample.mean(),
            "MSE_std": mse_per_sample.std(),
            "MAE_mean": mae_per_sample.mean(),
            "MAE_std": mae_per_sample.std(),
        }

        results.append(row)

    df = pd.DataFrame(results).sort_values("ROIs").reset_index(drop=True)

    # Save
    if save_name is not None:
        df.to_csv(save_name, index=False)
        print(f"Region-wise reconstruction performance saved to {save_name}")

    return df 

def split_train_val_test(x, train_percentage=0.7, validation_percentage=0.15, test_percentage=0.15, seed=42):
    """
    Docstring for split_train_val_test
    
    x: array, shape (n_trials, n_regions, n_timepoints)
        Before Rest(0:n_trials//2), then  Motor Imagery (n_trials//2:n_trials)
        They get NEGATIVE labels          They get POSITIVE labels
    """
    trp = train_percentage; vp = validation_percentage; tep = test_percentage

    if trp + vp + tep != 1:
        raise ValueError("Train, validation, and test percentages must sum to 1.")
    
    x = np.array(x)
    n_trials = x.shape[0]
    n_regions = x.shape[1]

    y = create_labels(n_trials, n_regions) 
    # shape (n_trials, n_regions) negative for Rest, positive for MI; numbered 1,2,3...n_regions per regions

    x = x.reshape(-1,x.shape[-1],) # shape (n_trials*n_regions, n_timepoints)
    y = y.reshape(-1,) # da eseguire insieme al reshape dei dati

    X_train, X_tmp, y_train, y_tmp = train_test_split(
    x, y,
    test_size=1-trp,          
    stratify=y,             # preserves label balance
    random_state=seed,
    shuffle=True
    )


    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=vp/(vp+tep),         
        stratify=y_tmp,     # preserves label balance
        random_state=seed+179,
        shuffle=True
    )

    return ((X_train, y_train),(X_val, y_val),(X_test, y_test))



####################################### OLD #############################################
"""
def plot_random_PowerSpectra_reconstructed( model, x_test, psd_test, freq_bins, n_samples=10, save_dir="reconstructed_spectra", random_state=224):
    
    Plot original, reconstructed, and filtered traces for random samples.

    Parameters
    ----------
    model : keras / tf model Trained autoencoder.
    x_test : np.ndarray  Shape: (n_samples, n_times)
    
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, ".gitignore"), "w") as f:
        f.write("*")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(x_test), size=n_samples, replace=False)


    lw = 2.3

    for k, idx in enumerate(indices):
        # --- reconstruction (EXACTLY like your snippet)
        x_in = x_test[idx:idx+1]
        power_rec = model(x_in)

        if isinstance(power_rec, dict):
            power_rec = power_rec["reconstruction"]
        power_rec = power_rec.numpy()

        plt.figure(figsize=(10, 4))
        plt.plot(freq_bins, psd_test[idx,:], linewidth=lw, label="original", color="C0")
        plt.plot(freq_bins, power_rec[0,:], linewidth=lw, label="reconstructed", color="C1")

        plt.title(f"Sample {idx}")
        plt.xlabel("Freq_bin")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(save_dir, f"Spectra_reconstruced_{k:02d}_idx{idx}.png")
        plt.savefig(fname, dpi=150)
        plt.close()




def evaluate_encoder_decoder_spectra_by_region(model, x_test, psd_test, y_test, save_name=None):

   # Compute reconstruction performance grouped by abs(y_test).


    # Reconstruct
    spectra_recon = model.predict(x_test)

    results = []

    for region in np.unique(np.abs(y_test)):
        mask = np.abs(y_test) == region
        spectra_true = psd_test[mask]
        spectra_pred = spectra_recon[mask]

        if len(spectra_true) == 0:
            continue
        
        axes = tuple(range(1, spectra_true.ndim))

        mse_per_sample = np.mean((spectra_true - spectra_pred) ** 2, axis=axes)
        mae_per_sample = np.mean(np.abs(spectra_true - spectra_pred), axis=axes)

        row = {
            "ROIs": region-1,
            "NROIs": len([region]),
            "n_windows": len(spectra_true),
            "MSE_mean": mse_per_sample.mean(),
            "MSE_std": mse_per_sample.std(),
            "MAE_mean": mae_per_sample.mean(),
            "MAE_std": mae_per_sample.std(),
        }

        results.append(row)

    df = pd.DataFrame(results).sort_values("ROIs").reset_index(drop=True)

    # Save
    if save_name is not None:
        df.to_csv(save_name, index=False)
        print(f"Region-wise reconstruction performance saved to {save_name}")

    return df







"""