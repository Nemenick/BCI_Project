import numpy as np
from mne.time_frequency import psd_array_welch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from mne.viz import Brain
import mne
import pandas as pd
import h5py
import os


def plot_brain(regions_to_color=[], atlas="aparc", printnameregions=False, brain_object_dict={}, showbrainplot=True, savepath=None, colors=[(1, 0, 0)]):
    mne.viz.set_3d_backend('pyvistaqt')  # or 'pyvistaqt'
    labels = mne.read_labels_from_annot("fsaverage", parc=atlas) # the Desikan-Killiany Atlas
    rtc = [i for i in regions_to_color]
    for i in range(len(rtc)):
        if type(rtc[i]) == int:
            rtc[i] = labels[rtc[i]].name
            if printnameregions:
                print(rtc[i])
    
    standard_dict = {
            "subject": "fsaverage",
            "surf": "pial",
            "hemi": "split", #split, both, lh,rh
            "background": "white",
            "views": ["dorsal"],
            "size": (600, 600),
            # Other parameters
            # "subjects_dir": subjects_dir,
            # "views": ["lat", "med"],
            # "offset": "auto",
            # "view_layout": "horizontal",
        }
        # 'dorsal', 'ventral', 'lat', 'medial', etc.)
        # views = 'medial'  # For example, 'lat' (lateral) view
        # brain.show_view(view=views)

    for key in standard_dict.keys():
        if key not in brain_object_dict:
            brain_object_dict[key] = standard_dict[key]
    brain_object_dict["show"] = showbrainplot

    brain = Brain(**brain_object_dict)

    if len(colors)==1:
        colors = np.repeat(colors, len(rtc), axis=0)
    elif len(colors)!= len(rtc):
        raise ValueError(f"Number of colors ({len(colors)}) must match number of regions_to_color ({len(rtc)}).")


    for i,region in enumerate(rtc):
        for label in labels:
            if label.name == region:
                brain.add_label(label, color=colors[i])  # Set color to redx

    if savepath is not None:
        brain.save_image(savepath)
    return brain

def read_subject(folder, DataType, condition:str, subject_index, verbose=True):
    """
    folder          : str, path to the folder containing the .mat files 
    DataType        : str, "EEG" or "MEG"     
    condition:str   : "MI" or "rest"                
    subject_index   : int, subject index (0-19)       
    verbose=True    
    """
    if condition.lower()=="rest" or condition.lower()=="baseline":
        condition = "Baseline"

    data = []
    
    filename=f"Data_{DataType}_{condition}_DK_V2.mat"
    path = f"{folder}{filename}"
    file = h5py.File(path, 'r')
    refs = file[filename[:-7]]
    # 96 trials, I read each trial
    for i in range (96):
        temp_trial = np.array(file[refs[i][subject_index]]).T
        if temp_trial.shape == (68, 1750):
            data.append(temp_trial)
        else:
            if verbose: print(f"subject_{subject_index} trial_{i} has wrong shape ({temp_trial.shape})")

    data=np.array(data)

    return data


def train_models(data_features, labels, n_kfold=5, scaler=StandardScaler, 
                 method=LinearDiscriminantAnalysis, seed=647,
                 verbose=False):
    
    kf = StratifiedKFold(
    n_splits=n_kfold,
    shuffle=True,
    random_state=seed
)

    fold_accuracies = []
    fold_confusions = []
    trained_models = []
    val_indexes = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data_features, labels)):
        if verbose:
            print(f"\n----- Fold {fold_idx+1} -----")

        X_train, X_val = data_features[train_idx], data_features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        val_indexes.append(val_idx)

        # StandardScaler:
        scaler_instance = scaler()
        X_train_scaled = scaler_instance.fit_transform(X_train)
        X_val_scaled   = scaler_instance.transform(X_val)

        # Model instance
        try:
            model = method(random_state=seed)
        except:
            model = method()
        model.fit(X_train_scaled, y_train)
        trained_models.append(model)

        # Predict on validation
        y_pred = model.predict(X_val_scaled)

        # Accuracy
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        fold_confusions.append(cm)
        if verbose:
            print(f"Fold {fold_idx+1} accuracy: {acc:.4f}, Confusion matrix:")
            print(cm)

    # Summary
    if verbose:
        print("\n==================================")
        print("Mean accuracy:", np.mean(fold_accuracies))
        print("Std accuracy:", np.std(fold_accuracies))
    return fold_accuracies, fold_confusions, trained_models,val_indexes

def saveresults_pickle(new_rows, inputfile=None, outputfile=None, backup=True, resetindex=True):
    # filepath "/Users/giovanni.messuti/Desktop/BCI_Project/Results/BCI_Performances.pkl"
    """
    if inputfile is a valid pickle file, read it and append the new_rows to it.
    If not, create a new pickle file with new_rows.
    If outputfile is None, save to inputfile.
    If both inputfile and outputfile are None, save to a default file
    If backup is True, create a backup of the inputfile before modifying it.
    """

    if outputfile is None:
        outputfile = inputfile
    if outputfile is None:
        outputfile = "BCI_Performances.pkl"
        print(f"No input or output file specified, saving to a default file")

    while outputfile != inputfile and os.path.exists(outputfile):
        print(f"Output file {outputfile} is different from Input file and already exists. adding extention '_new' to the output file name.")
        outputfile = outputfile + "_new"

    try:
        results = pd.read_pickle(inputfile)
        if backup:
            results.to_pickle(inputfile+"Backup")
        results=pd.concat((results,pd.DataFrame(new_rows)))
        if resetindex:
            results.reset_index(drop=True,inplace=True)
        index_of_duplicates = results[results.duplicated(subset=results.keys().drop(["Date","ROIs","Performance"]))].index.values
        print(f"There are {len(index_of_duplicates)} duplicated items (based on all columns except Date, ROIs, Performance).\n", 
              f"Index of the duplicated items: {index_of_duplicates}")
        results.to_pickle(outputfile)

    except Exception as e:
        print("#######"*10,"\n","Exception occurred:", e,"\n","#######"*10)
        print(f"Cannot Read a previous file {inputfile}, creating a new one at {outputfile}")
        if resetindex:
            new_rows.reset_index(drop=True,inplace=True)
        new_rows.to_pickle(outputfile+"new") 

def select_rows(dataframe, conditions):
    """
    Return the rows of the dataframe that match all the conditions specified in the conditions dictionary.
    dataframe: pd.DataFrame
    conditions: dict, where keys are column names and values are the desired values to filter on.
    """
    mask = pd.Series(True, index=dataframe.index)
    for col, val in conditions.items():
        if col not in dataframe.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        
        mask &= (dataframe[col] == val)
    return dataframe[mask]

def compute_welch_select_regions(data_MI=None, data_Rest=None, sfreq=250, starttime=250, endtime=250, nbest_regions=8, 
                    wpsdMI=None, wpsdRest=None, frequiMI=None, select_regions=True,
                    kargs_welch={"fmin":2, "fmax":45,"n_per_seg":250, "n_fft":300,"n_overlap":125, "verbose":False},
                    important_regions=[4, 5, 32, 33, 44, 45, 48, 49],):
    """
    Can compute spectra and / or select regions (on computed spectra or on provided spectra)
    For the nbest_regions ROIs (where i make sure to include the important regions).
    Selected are the regions where the max(Rest-MI) / max(Rest) has the max values.
    ----------
    - data_MI, data_Rest: (n_trials x n_ROIs x n_samples).
    - sfreq: sampling frequency
    - starttime, endtime : in secs, Segment boundaries to apply to data.
    - min_freq_frature, max_freq_frature:
        Frequency limits for feature extraction (where to compute mean and max)
    - nbest_regions : Number of most informative regions to select.
    - kargs_welch : dict, Parameters for scipy.signal.welch()
        kargs = {"sfreq":sfreq, "fmin":fmin, "fmax":fmax,
                "n_per_seg":n_per_seg, "n_fft":n_fft,
                "n_overlap":n_overlap}
    - important regions: Regions to be included for sure
    
    Returns
    -------
    - selected_regions
    - welch spectra of the data: (n_trials x n_ROIs x n_freqs).
    """
    try:
        ntotROIs = data_MI.shape[1]
    except:
        ntotROIs = wpsdMI.shape[1]
    #data.shape (trials, ROIs, timepoints)
    returned_objects = []




    if wpsdMI is None:
        # vedo se ho tutte le info che servono
        standard_dict = {"fmin":2, "fmax":45,"n_per_seg":250, "n_fft":300,"n_overlap":125, "verbose":False}
        for key in standard_dict.keys():
            if key not in kargs_welch:
                kargs_welch[key] = standard_dict[key]
        # procedo 
        stime = 1 * starttime
        etime = 1 * endtime
        kargs_welch["sfreq"]=sfreq
        ######################## Compute Power Spectra ########################
        wpsdMI, frequiMI = psd_array_welch(data_MI[:,:,stime:-etime], **kargs_welch )
        wpsdRest, frequiRest = psd_array_welch(data_Rest[:,:,stime:-etime], **kargs_welch)
        # wpsd.shape (trials, ROIs, freqs)
        returned_objects.append(wpsdMI); returned_objects.append(wpsdRest); returned_objects.append(frequiMI); 

    if select_regions:
        ######################## Search the best regions ########################
        # (regions where the the %difference at the freq with max difference is the highest)
        # I choose the nregions where I have the greatest difference between MI and Rest 
        # (but being careful to include the Interesting_regions)
        max_diff_perc = []
        max_rest = []
        for roi in range(ntotROIs):
            wpsdMI_1ROI =wpsdMI[:,roi,:]
            wpsdRest_1ROI =wpsdRest[:,roi,:]
            wpsd_MI_mean = wpsdMI_1ROI.mean(axis=0)
            wpsd_Rest_mean = wpsdRest_1ROI.mean(axis=0)
            
            max_rest.append(np.max(wpsd_Rest_mean))
            max_diff_perc.append(np.max(wpsd_Rest_mean-wpsd_MI_mean)/max_rest[-1])

        selected_regions = [i for i in important_regions]
        for i in range(ntotROIs):
            if len(selected_regions) >= nbest_regions:
                break
            if np.argsort(max_diff_perc)[-i-1] not in selected_regions:
                selected_regions.append(int(np.argsort(max_diff_perc)[-i-1]))

        if returned_objects == []:
            returned_objects = selected_regions
        else:
            returned_objects.append(selected_regions)
    
    return returned_objects # [wpsdMI, wpsdRest, frequiMI, selected_regions]
        
def extract_features_from_spectra_more_bands(wpsdMI, wpsdRest, selected_regions, frequiMI,
                    min_freq_frature=[4,8,12], max_freq_frature=[8,12,30], 
                    select_mean=True, select_max=True):
    """
    Extract frequency-domain features from MI and Rest spectra of data.
    Features are the mean and max in a fixed window of freqs (min_freq_frature, max_freq_frature)
    For the selected ROIs. Selected previously
    ----------
    
    - welch spectra of the data: (n_trials x n_ROIs x n_freqs)
    - selected_regions
    - frequencies corresponding to the bins in welch spectra (n_freqs)

    Returns
    -------
    - features_MI, - features_Rest: mean and max of each selected ROI, 
        computed in the freq interval (min_freq_frature,max_freq_frature)
    - labels_MI, - labels_Rest (0 for Rest and 1 for MI)
    - selected_regions
    """
    if isinstance(min_freq_frature, int) or isinstance(min_freq_frature, float):
        min_freq_frature = [min_freq_frature]
    if isinstance(max_freq_frature, int) or isinstance(max_freq_frature, float):
        max_freq_frature = [max_freq_frature]

    nTrials_MI = wpsdMI.shape[0]
    nTrials_Rest = wpsdRest.shape[0]
        
    # selecting the freq range on wich compute features 
    # as features i chose the max and the maen of selected_ROIs 
    # in freqs (min_freq_frature, min_freq_frature)

    ######################## Extract the features ########################

    start_freq_idx = [np.where(frequiMI>=min_freq_frature[i])[0][0] for i in range(len(min_freq_frature))]
    end_freq_idx = [np.where(frequiMI<=max_freq_frature[i])[0][-1] for i in range(len(max_freq_frature))]
    features_MI = []
    features_Rest = []
    for _ in range(len(min_freq_frature)):

        wpsdRest_ROIs_freqs = wpsdRest[:,selected_regions,start_freq_idx[_]:end_freq_idx[_]+1]
        wpsdMI_ROIs_freqs = wpsdMI[:,selected_regions,start_freq_idx[_]:end_freq_idx[_]+1]
        # wpsdMI_ROIs_freqs.shape (trials, selected_ROIs, selected_Freqs)

        if select_mean:
            features_MI.append(wpsdMI_ROIs_freqs.mean(axis=-1,keepdims=True))
            features_Rest.append(wpsdRest_ROIs_freqs.mean(axis=-1,keepdims=True))

        if select_max:  
            features_MI.append(wpsdMI_ROIs_freqs.max(axis=-1,keepdims=True))
            features_Rest.append(wpsdRest_ROIs_freqs.max(axis=-1,keepdims=True))
    
    features_MI = np.concatenate(features_MI,axis=-1)
    features_Rest = np.concatenate(features_Rest,axis=-1)
    # features_MI.shape (Trials, selected_ROIs,6)

    features_MI = features_MI.reshape(nTrials_MI,-1)
    features_Rest = features_Rest.reshape(nTrials_Rest,-1)

    labels_MI = np.array([1 for _ in range(nTrials_MI)])
    labels_Rest = np.array([0 for _ in range(nTrials_Rest)])

    return features_MI, features_Rest, labels_MI, labels_Rest, selected_regions

# deprecated 
# but still in use
def extract_features_more_bands(data_MI, data_Rest, sfreq, starttime, endtime, nbest_regions,
                    min_freq_frature=[4,8,12], max_freq_frature=[8,12,30], 
                    kargs_welch={"fmin":2, "fmax":45,"n_per_seg":250, "n_fft":300,"n_overlap":125},
                    important_regions=[4, 5, 32, 33, 44, 45, 48, 49],
                    select_mean=True, select_max=True):
    """
    Extract frequency-domain features from MI and Rest data.
    Features are the mean and max in a fixed window of freqs (min_freq_frature, max_freq_frature)
    For the nbest_regions ROIs (where i make sure to include the important regions).
    These are the regions where the max(Rest-MI) / max(Rest) has the max values.
    
    ----------
    - data_MI, data_Rest: (n_trials x n_ROIs x n_samples).
    - sfreq: sampling frequency
    - starttime, endtime : in secs, Segment boundaries to apply to data.
    - min_freq_frature, max_freq_frature:
        Frequency limits for feature extraction (where to compute mean and max)
    - nbest_regions : Number of most informative regions to select.
    - kargs_welch : dict, Parameters for scipy.signal.welch()
        kargs = {"sfreq":sfreq, "fmin":fmin, "fmax":fmax,
                "n_per_seg":n_per_seg, "n_fft":n_fft,
                "n_overlap":n_overlap}
    - important regions: Regions to be included for sure
    

    Returns
    -------
    - features_MI, - features_Rest: mean and max of each selected ROI, 
        computed in the freq interval (min_freq_frature,max_freq_frature)
    - labels_MI, - labels_Rest (0 for Rest and 1 for MI)
    - selected_regions
    """

    if isinstance(min_freq_frature, int) or isinstance(min_freq_frature, float):
        min_freq_frature = [min_freq_frature]
    if isinstance(max_freq_frature, int) or isinstance(max_freq_frature, float):
        max_freq_frature = [max_freq_frature]

    ntotROIs = data_MI.shape[1]
    nTrials_MI = data_MI.shape[0]
    nTrials_Rest = data_Rest.shape[0]
    #data.shape (trials, ROIs, timepoints)

    # vedo se ho tutte le info che servono
    standard_dict = {"fmin":2, "fmax":45,"n_per_seg":250, "n_fft":300,"n_overlap":125, "verbose":False}
    for key in standard_dict.keys():
        if key not in kargs_welch:
            kargs_welch[key] = standard_dict[key]
    # procedo 
    stime = 1 * starttime
    etime = 1 * endtime
    kargs_welch["sfreq"]=sfreq

    ######################## Compute Power Spectra ########################
    wpsdMI, frequiMI = psd_array_welch(data_MI[:,:,stime:-etime], **kargs_welch )
    wpsdRest, frequiRest = psd_array_welch(data_Rest[:,:,stime:-etime], **kargs_welch)

    ######################## Search the best regions ########################

    # (regions where the the %difference at the freq with max difference is the highest)
    # I choose the nregions where I have the greatest difference between MI and Rest 
    # (but being careful to include the Interesting_regions)
    max_diff_perc = []
    max_rest = []
    for roi in range(ntotROIs):
        wpsdMI_1ROI =wpsdMI[:,roi,:]
        wpsdRest_1ROI =wpsdRest[:,roi,:]
        wpsd_MI_mean = wpsdMI_1ROI.mean(axis=0)
        wpsd_Rest_mean = wpsdRest_1ROI.mean(axis=0)
        
        max_rest.append(np.max(wpsd_Rest_mean))
        max_diff_perc.append(np.max(wpsd_Rest_mean-wpsd_MI_mean)/max_rest[-1])
        

    selected_regions = [i for i in important_regions]
    for i in range(ntotROIs):
        if len(selected_regions) >= nbest_regions:
            break
        if np.argsort(max_diff_perc)[-i-1] not in selected_regions:
            selected_regions.append(int(np.argsort(max_diff_perc)[-i-1]))
        

    # selecting the freq range on wich compute features 
    # as features i chose the max and the maen of selected_ROIs 
    # in freqs (min_freq_frature, min_freq_frature)

    ######################## Extract the features ########################

    start_freq_idx = [np.where(frequiMI>=min_freq_frature[i])[0][0] for i in range(len(min_freq_frature))]
    end_freq_idx = [np.where(frequiMI<=max_freq_frature[i])[0][-1] for i in range(len(max_freq_frature))]
    features_MI = []
    features_Rest = []
    for _ in range(len(min_freq_frature)):

        wpsdRest_ROIs_freqs = wpsdRest[:,selected_regions,start_freq_idx[_]:end_freq_idx[_]+1]
        wpsdMI_ROIs_freqs = wpsdMI[:,selected_regions,start_freq_idx[_]:end_freq_idx[_]+1]
        # wpsdMI_ROIs_freqs.shape (trials, selected_ROIs, selected_Freqs)

        if select_mean:
            features_MI.append(wpsdMI_ROIs_freqs.mean(axis=-1,keepdims=True))
            features_Rest.append(wpsdRest_ROIs_freqs.mean(axis=-1,keepdims=True))

        if select_max:  
            features_MI.append(wpsdMI_ROIs_freqs.max(axis=-1,keepdims=True))
            features_Rest.append(wpsdRest_ROIs_freqs.max(axis=-1,keepdims=True))
    
    features_MI = np.concatenate(features_MI,axis=-1)
    features_Rest = np.concatenate(features_Rest,axis=-1)
    # features_MI.shape (Trials, selected_ROIs,6)

    features_MI = features_MI.reshape(nTrials_MI,-1)
    features_Rest = features_Rest.reshape(nTrials_Rest,-1)

    labels_MI = np.array([1 for i in range(nTrials_MI)])
    labels_Rest = np.array([0 for i in range(nTrials_Rest)])

    return features_MI, features_Rest, labels_MI, labels_Rest, selected_regions


""" But not used anymore
    def extract_features(data_MI, data_Rest, sfreq, starttime, endtime,
                        min_freq_frature, max_freq_frature, nbest_regions, 
                        kargs_welch={"fmin":2, "fmax":45,"n_per_seg":250, "n_fft":300,"n_overlap":125},
                        important_regions=[4, 5, 32, 33, 44, 45, 48, 49]):
        
        # Extract frequency-domain features from MI and Rest data.
        # Features are the mean and max in a fixed window of freqs (min_freq_frature, max_freq_frature)
        # For the nbest_regions ROIs (where i make sure to include the important regions).
        # These are the regions where the max(Rest-MI) / max(Rest) has the max values.
        
        # ----------
        # - data_MI, data_Rest: (n_trials x n_ROIs x n_samples).
        # - sfreq: sampling frequency
        # - starttime, endtime : in secs, Segment boundaries to apply to data.
        # - min_freq_frature, max_freq_frature:
        #     Frequency limits for feature extraction (where to compute mean and max)
        # - nbest_regions : Number of most informative regions to select.
        # - kargs_welch : dict, Parameters for scipy.signal.welch()
        #     kargs = {"sfreq":sfreq, "fmin":fmin, "fmax":fmax,
        #             "n_per_seg":n_per_seg, "n_fft":n_fft,
        #             "n_overlap":n_overlap}
        # - important regions: Regions to be included for sure
        

        # Returns
        # -------
        # - features_MI, - features_Rest: mean and max of each selected ROI, 
        #     computed in the freq interval (min_freq_frature,max_freq_frature)
        # - labels_MI, - labels_Rest (0 for Rest and 1 for MI)
        # - selected_regions

        ntotROIs = data_MI.shape[1]
        nTrials_MI = data_MI.shape[0]
        nTrials_Rest = data_Rest.shape[0]
        #data.shape (trials, ROIs, timepoints)

        stime = 1 * starttime
        etime = 1 * endtime
        kargs_welch["sfreq"]=sfreq

        wpsdMI, frequiMI = psd_array_welch(data_MI[:,:,stime:-etime], **kargs_welch )
        wpsdRest, frequiRest = psd_array_welch(data_Rest[:,:,stime:-etime], **kargs_welch)
        # wpsdMI.shape (trials, ROIs, freqs)

        # search the best regions 
        # (regions where the the %difference at the freq with max difference is the highest)
        # I choose the nregions where I have the greatest difference between MI and Rest 
        # (but being careful to include the Interesting_regions)
        max_diff_perc = []
        max_rest = []
        for roi in range(ntotROIs):
            wpsdMI_1ROI =wpsdMI[:,roi,:]
            wpsdRest_1ROI =wpsdRest[:,roi,:]
            wpsd_MI_mean = wpsdMI_1ROI.mean(axis=0)
            wpsd_Rest_mean = wpsdRest_1ROI.mean(axis=0)
            
            max_rest.append(np.max(wpsd_Rest_mean))
            max_diff_perc.append(np.max(wpsd_Rest_mean-wpsd_MI_mean)/max_rest[-1])
            

        selected_regions = [i for i in important_regions]
        for i in range(ntotROIs):
            if len(selected_regions) >= nbest_regions:
                break
            if np.argsort(max_diff_perc)[-i-1] not in selected_regions:
                selected_regions.append(int(np.argsort(max_diff_perc)[-i-1]))
            

        # selecting the freq range on wich compute features 
        # as features i chose the max and the maen of selected_ROIs 
        # in freqs (min_freq_frature, min_freq_frature)
        start_freq_idx = np.where(frequiMI>=min_freq_frature)[0][0]
        end_freq_idx = np.where(frequiMI<=max_freq_frature)[0][-1]

        wpsdRest_ROIs_freqs = wpsdRest[:,selected_regions,start_freq_idx:end_freq_idx]
        wpsdMI_ROIs_freqs = wpsdMI[:,selected_regions,start_freq_idx:end_freq_idx]
        # wpsdMI_ROIs_freqs.shape (trials, selected_ROIs, selected_Freqs)

        features_MI = np.concatenate((wpsdMI_ROIs_freqs.mean(axis=-1,keepdims=True),
                                    wpsdMI_ROIs_freqs.max(axis=-1,keepdims=True)),
                                    axis=-1)
        # features_MI.shape (Trials, selected_ROIs, 2)
        features_MI = features_MI.reshape(nTrials_MI,-1)

        features_Rest = np.concatenate((wpsdRest_ROIs_freqs.mean(axis=-1,keepdims=True),
                                    wpsdRest_ROIs_freqs.max(axis=-1,keepdims=True)),
                                    axis=-1)
        features_Rest = features_Rest.reshape(nTrials_Rest,-1)

        labels_MI = np.array([1 for i in range(nTrials_MI)])
        labels_Rest = np.array([0 for i in range(nTrials_Rest)])

        return features_MI, features_Rest, labels_MI, labels_Rest, selected_regions


        

    def read_subject(file, refs, subject_index, verbose=True):
    data = []
    
    # 96 trials, I read each trial
    for i in range (96):
        temp_trial = np.array(file[refs[i][subject_index]]).T
        if temp_trial.shape == (68, 1750):
            data.append(temp_trial)
        else:
            if verbose: print(f"subject_{subject_index} trial_{i} has wrong shape ({temp_trial.shape})")

    data=np.array(data)
    return data



    """