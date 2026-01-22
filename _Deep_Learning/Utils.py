import numpy as np
import scipy.signal as sc_sig
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split


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
    """
    trp = train_percentage; vp = validation_percentage; tep = test_percentage

    if trp + vp + tep != 1:
        raise ValueError("Train, validation, and test percentages must sum to 1.")
    
    x = np.array(x)
    n_trials = x.shape[0]
    n_regions = x.shape[1]

    y = create_labels(n_trials, n_regions) 
    # shape (n_trials, n_regions) positive for MI, negative for Rest; numbered 1,2,3...n_regions per regions

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
            print(X_region.shape, mu.shape, std.shape)
            print(std)

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



