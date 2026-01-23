import numpy as np
import scipy.signal as sc_sig
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################# Simple functions #############################

def freq_filter(signal,sf,freqs,type_filter="bandpass", order_filter=4):
    """ 
    sf: sampling rate of the input waveform
    freqs: list of frequences (e.g. 2 for bandpass), or single float (e.g. for highpass)
        sf sampling frequence  """
    # type_filter: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’

    freqs=np.array(freqs)
    filt_b1,filt_a1=sc_sig.butter(order_filter,freqs/(sf/2),btype=type_filter)
    filtered_sig=sc_sig.filtfilt(filt_b1,filt_a1,sc_sig.detrend(signal)) # both detrend and filtfilt have axis = -1 by default OK!
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



def create_labels(n_trials, n_regions=68):

    # assegno negativi a quelli di prima (Rest) e positivi a quelli di dopo (MI) 
    y = np.ones((n_trials, n_regions))
    y[0:n_trials//2, :] = -1

    # assegno valore in base a ROI, da ±1 a ±68
    for col in range(y.shape[1]):
        y[:, col] *= (col+1)
    return y


############################ Composite functions #############################


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

    y = y.reshape(-1,) # da eseguire insieme al reshape dei dati
    x = x.reshape(-1,x.shape[-1],) # shape (n_trials*n_regions, n_timepoints)

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
        random_state=seed,
        shuffle=True
    )

    return ((X_train, y_train),(X_val, y_val),(X_test, y_test))


def windowize(X,Y, win_len=128, shift=62):
    """
    Function to extract windows from X and replicate labels Y accordingly 
    to assign the same label to all windows extracted from a single trace (trial)
    
    X: (n_traces, n_timepoints)
    Y: (n_traces,)
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


import numpy as np

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

    def __call__(self, X):
        if self.method == "zscore":
            return self.zscore_normalizer(X)
        elif self.method == "minmax":
            return self.minmax_normalizer(X)
        else:
            raise ValueError("Unknown method")


############################## Saving results #############################

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