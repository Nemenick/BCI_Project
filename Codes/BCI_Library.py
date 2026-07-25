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
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import tensorflow as tf

# Important_regions_right =   [5, 33, 45, 49]
# Important_regions_left =   [4, 32, 44, 48]


################################## Random utils ######################################################

def plot_brain(regions_to_color=[], atlas="aparc", printnameregions=False, brain_object_dict={}, 
               showbrainplot=True, savepath=None, colors=[(1, 0, 0)]):
    
    mne.viz.set_3d_backend('notebook')  # or 'pyvistaqt'
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

def select_rows(dataframe, conditions:dict):
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

def read_ranks_subect(df_path, subject, read_all_trials=False):
    """
    -   df of df_path structure:
    Subject, rois_fold_1 (regions sorted by rank in dold 1), P-value (or effect size) (optional), fold_2, fold_3...
    -   subject: index of the subject to be read
    -   read_all_trials: If read not a fold, the column "ROIs_All_trials", containing the selection made on all the data, not dividing by folds 
    
    reads the ranks of that subject among folds from a saved dataframe
    """

    piccolo = "rois_fold_" if read_all_trials == False else "rois_all_t"

    pandadizio=pd.read_pickle(df_path)
    rankings = []
    for fold in range(1,len(pandadizio.keys())):
        if pandadizio.iloc[:,fold].name[:10].lower() == piccolo:
            rankings.append(list(pandadizio.iloc[subject,fold][:]))
    return np.array(rankings)

def plot_violin_all_subj(Dataframe_performances,  num_settings_showed,  dict_fixed_selection,  dict_per_setting,
                  x_ticklabels,  textes,  textes_x_positions,
                  divisioni, xlabel="", hlines_positions=[],
                  track_subjects=True,  savepath=False, figsize=(8,5), additional_title=""):
    """
    divisioni: [0,3,6,9] - dove andrò a mettere le vlines e dove farò il track dei soggetti (linee tratteggiate tra i diversi violin)
    """
    chaanche_lvl = 0.58 # https://pubmed.ncbi.nlm.nih.gov/25596422/
    hlines_positions = hlines_positions + [chaanche_lvl]
    PerData = Dataframe_performances
    vlines_positions=[divisioni[_]-0.5 for _ in range(1,len(divisioni)-1)]
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    nps = num_settings_showed
    x_positions = np.arange(nps)

    styles=["--","solid"]
    markes=["s","o"]
    size=20

    all_perfs = []
    for subject in range(20):

        perfs_per_subject = []
        for setting in range(nps):
            # Extract performances
            tmp_dict = {"subject":subject}
            for key in dict_per_setting:
                tmp_dict[key] = dict_per_setting[key][setting]
            rows = select_rows(PerData, dict_fixed_selection | tmp_dict)
            perfs_per_subject.append(np.array(list(rows["Accuracy"].values)).mean())
        all_perfs.append(perfs_per_subject)
        shifcolor=1 if subject>=10 else 0
        color=f"C{subject+shifcolor}"

        ax.scatter(x_positions, perfs_per_subject, marker=markes[subject//10],s=size,color=color)#facecolors='none',edgecolors=color

        for __ in range(len(divisioni)-1):
            if not track_subjects: break
            ax.plot(x_positions[divisioni[__]:divisioni[__+1]], perfs_per_subject[divisioni[__]:divisioni[__+1]], linestyle=styles[subject//10], linewidth=0.7,color=color)

        
    all_perfs = np.array(all_perfs)   # shape (20 subjects, 9 settings)

    #Violin plot
    ax_violin = ax
    violins = ax_violin.violinplot(
        all_perfs,
        positions=x_positions,
        showmeans=True,
        #showmedians=True,
        widths=0.8,
    )
    violins['cmeans'].set_color('red') # mean line vp['cmedians'].set_color('black') # median line

    for part in violins['bodies']+[violins['cbars']]:
        part.set_zorder(-20)
    xlims = ax.get_xlim()
    ax_violin.set_ylabel("Accuracy")
    ax_violin.set_xlabel(xlabel)
    ax.hlines(hlines_positions, -1, num_settings_showed, colors='grey', linestyles='dashed', label="Chance Level",zorder=-1, alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_ticklabels)
    ax.vlines(vlines_positions, 0.5, 1, colors='gray', linestyles='dashed')
    ax.set_ylim(0.5, 1)
    for __ in range(len(textes)):
        ax.text(textes_x_positions[__], 0.1, textes[__], transform=ax.transAxes, fontsize=12, verticalalignment='top',horizontalalignment='center')

    ax.text(1., 0.192, f"Chance level  ", transform=ax.transAxes, fontsize=9, verticalalignment='top',horizontalalignment='right')
    title = f"Distribution of Accuracy Across Subjects for Each Setting\n"
    for key,value in dict_fixed_selection.items():
        title+= f"{key}:{value} | "
    title=title[:-2]
    title += additional_title
    ax.set_title(title, fontsize=13)
    ax.set_xlim(xlims)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath,dpi=300)
    plt.show()
    return fig, ax
    
def plot_violin_track_single_fold(PerformanceData,num_settings_showed_per_subject,
                                 dict_fixed_selection,dict_per_setting,
                                 x_ticklabels,hlines_positions = [], savepath=False, additional_title="", vlines_positions = None):
    size = 20
    chance_lvl = 0.58 # https://pubmed.ncbi.nlm.nih.gov/25596422/
    hlines_positions = hlines_positions + [chance_lvl]
    x_positions = np.arange(num_settings_showed_per_subject)

    num_subjects = 20
    n_rows = 5
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(20, 16),
        sharex=True,
        sharey=True
    ); axes = axes.flatten()

    # ==========================
    # LOOP OVER SUBJECTS
    for subject in range(num_subjects):
        ax = axes[subject]
        perfs_subject = []
        for setting in range(num_settings_showed_per_subject):
            tmp_dict = {"subject": subject}
            for key in dict_per_setting:
                tmp_dict[key] = dict_per_setting[key][setting]

            rows = select_rows(PerformanceData, dict_fixed_selection | tmp_dict)
            perfs_subject.append(
                np.array(list(rows["Accuracy"].values))
            )

        # ----------------------
        # Violin plot
        # ----------------------
        perfs_subjectbp = [np.asarray(p).ravel() for p in perfs_subject]

        vp = ax.violinplot(
            perfs_subjectbp,
            positions=x_positions,
            widths=0.5,
            showmeans=True
        )

        for body in vp['bodies']: 
            body.set_facecolor("lightgray")
            body.set_edgecolor("black") 
            body.set_alpha(0.6)

        x_scatter = np.concatenate([ np.repeat(x_positions[i], len(perfs_subject[i][0])) for i in range(len(perfs_subject)) ])
        y_scatter = np.concatenate([i[0] for i in perfs_subject], axis=0)
        ax.scatter( x_scatter, y_scatter, s=size, alpha=0.7, zorder=10, color="red")
        xmin, xmax = ax.get_xlim()
        # CHANCE LEVEL
        ax.hlines(hlines_positions, xmin, xmax, colors="gray", linestyles="dashed", alpha=0.7)
        if vlines_positions:
            ax.vlines(vlines_positions,0,1, colors="gray", linestyles="dashed", alpha=0.7)
        ax.set_title(f"Subject {subject}", fontsize=10)

    # ==========================
    # AXIS FORMATTING
    for ax in axes[-n_cols:]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_ticklabels)
    for ax in axes[::n_cols]:
        ax.set_ylabel("Accuracy")
    axes[0].set_ylim(0.5, 1.05);  ax.set_xlim(-0.4, num_settings_showed_per_subject-0.6)
    # Remove unused axes if any
    for ax in axes[num_subjects:]:
        ax.axis("off")
    # ==========================
    # TITLE
    title = "Accuracy Distribution per Subject\n"
    for k, v in dict_fixed_selection.items():
        title += f"{k}:{v} | "

    fig.suptitle(title[:-2]+additional_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        plt.savefig(savepath,dpi=300)
    plt.show()


################################## Trainings / models ######################################################

def build_compiled_mlp(input_dim, hidden_layers=[20], activation='leaky_relu', output_dim=1, output_activation='sigmoid', **adam_kargs):
    """
    Build a simple MLP model using Keras.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))

    model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))

    adam = tf.keras.optimizers.Adam(**adam_kargs)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_models(data_features, labels, n_kfold=5, scaler=StandardScaler, 
                 method=LinearDiscriminantAnalysis, seed=224,
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

def train_single_model(data_features, labels, train_idx, val_idx, method=LinearDiscriminantAnalysis, 
                 scaler=StandardScaler, seed=224, return_predictions=False,
                 verbose=False, fit_kwargs={}):
    
    X_train, X_val = data_features[train_idx], data_features[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    
    # StandardScaler:
    if scaler is not None:
        scaler_instance = scaler()
        X_train_scaled = scaler_instance.fit_transform(X_train)
        X_val_scaled   = scaler_instance.transform(X_val)

    # Model instance
    try:
        model = method(random_state=seed)
    except:
        try:
            model = method()
        except:
            model = method
    is_keras_model = isinstance(model, tf.keras.Model)

    if is_keras_model:
        model.fit(X_train_scaled, y_train,validation_data=(X_val_scaled, y_val), **fit_kwargs)
    else:
        model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_val_scaled)
    
    if is_keras_model:
        y_pred = (y_pred > 0.5).astype(int).ravel()

    # Accuracy   # Confusion matrix
    acc = accuracy_score(y_val, y_pred)
    confusion_mat = confusion_matrix(y_val, y_pred)

    if verbose:
        print(confusion_mat)
    if return_predictions:
        return acc, confusion_mat, model, {"y_true":y_val, "y_pred":y_pred}
    return acc, confusion_mat, model

def saveresults_pickle(new_rows, inputfile=None, outputfile=None, backup=True, resetindex=True):
    # filepath "../Results/BCI_Performances.pkl"
    """
    if inputfile is a valid pickle file, read it and append the new_rows to it.
    If not, create a new pickle file with new_rows.
    If outputfile is None, save to inputfile.
    If both inputfile and outputfile are None, save to a default file
    If backup is True, create a backup of the inputfile before modifying it.
    """

    """new_row = {
    "ROIs": [selected_regions_EEG],
    "NROIs":[25]
    "DataType": ["EEG"],
    "freqs_band": [((4,8),(8,12),(12,30))],  
    "subject": [subject_index],
    "Classifier": ["LDA"],
    "Performance": [training[0]],
    "Features": ["PowerSpectra - MeanMax band"],
    "Comments": ["-"],
    "Date": [today],
    "Personalized_Freq_Range" = False,
}
"""
    
    if outputfile is None:
        outputfile = inputfile
    if outputfile is None:
        outputfile = "BCI_Performances_default.pkl"
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
        index_of_duplicates = results[results.duplicated(subset=results.keys().drop(["Date","ROIs","Accuracy"]))].index.values
        print(f"There are {len(index_of_duplicates)} duplicated items (based on all columns except Date, ROIs, Accuracy).\n", 
              f"Index of the duplicated items: {index_of_duplicates}")
        results.to_pickle(outputfile)
        print(f"\nResults saved at: {outputfile}")

    except Exception as e:
        print("#######"*10,"\n","Exception occurred:", e,"\n","#######"*10)
        print(f"Cannot Read a previous file {inputfile}, creating a new one at {outputfile}")
        if resetindex:
            new_rows.reset_index(drop=True,inplace=True)
        new_rows.to_pickle(outputfile+"new") 
        print(f"\nResults saved at: {outputfile+"new"}")
        


################################## Features extraction - selection ##################################

def cohens_d_per_columm(X_MI, X_Re, return_all=False):
    """
    Compute Cohen's d - effect size - at each column between two sets of series.
    X,Y : ndarray, shape (n_samples, columns)
    Returns -------
    d_max : float Maximum absolute Cohen's d across timepoints

    verified: same as pingouin.compute_effsize(x, y, paired=True, eftype='cohen')
    """

    mean_X_MI = X_MI.mean(axis=0)
    mean_X_Re = X_Re.mean(axis=0)

    var_X_MI = X_MI.var(axis=0, ddof=1)
    var_X_Re = X_Re.var(axis=0, ddof=1)

    pooled_std = np.sqrt((var_X_Re + var_X_MI) / 2)
    pooled_std[pooled_std == 0] = np.nan
    # d_t : ndarray, shape (columns,) Cohen's d at each column
    d_t = (mean_X_Re - mean_X_MI) / pooled_std
    d_max = np.nanmax(d_t)

    if return_all:
        return d_t
    else:
        return d_max

def compute_welch(data_MI, data_Rest, sfreq=250, starttime=250, endtime=250,
                    kargs_welch={"fmin":8, "fmax":30,"n_per_seg":250, "n_fft":300,"n_overlap":125, "verbose":False}):
    """
    Compute spectra 
    ----------
    - data_MI, data_Rest: (n_trials x n_ROIs x n_samples).
    - sfreq: sampling frequency
    - starttime, endtime : in samples, Segment boundaries to apply to data.
    - kargs_welch : dict, Parameters for scipy.signal.welch()
        kargs = {"sfreq":sfreq, "fmin":fmin, "fmax":fmax,
                "n_per_seg":n_per_seg, "n_fft":n_fft,
                "n_overlap":n_overlap}
    ----------
    Returns
    - welch spectra of the data: (n_trials x n_ROIs x n_freqs).
    """
    #data.shape (trials, ROIs, timepoints)
    returned_objects = []

    # vedo se ho tutte le info che servono
    standard_dict = {"fmin":8, "fmax":30,"n_per_seg":250, "n_fft":300,"n_overlap":125, "verbose":False}
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

    return returned_objects # [wpsdMI, wpsdRest, frequiMI]
        
def select_regions_MyCriteria(wpsdMI, wpsdRest,
                    nbest_regions=8, 
                    important_regions=[4, 5, 32, 33, 44, 45, 48, 49]):
    """
    Select regions (on provided spectra)
    wpsdMI.shape = (trials, ROIs, frequency_bins)
    For the nbest_regions ROIs (where i make sure to include the important regions).
    Selected are the regions where the max(Rest-MI) / max(Rest) has the max values.
    ----------

    - nbest_regions : Number of most informative regions to select.
    - important regions: Regions to be included for sure
    
    Returns - selected_regions: Array
    """
    ######################## Search the best regions ########################
    # (regions where the the %difference at the freq with max difference is the highest)
    # I choose the nregions where I have the greatest difference between MI and Rest 
    # (but being careful to include the Interesting_regions)
    ntotROIs = wpsdMI.shape[1]
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

    return selected_regions

def select_regions_Cohen_effect_size(wpsdMI, wpsdRest,
                    nbest_regions=8, 
                    important_regions=[]):
    """
    Select regions (on provided spectra)
    wpsdMI.shape = (trials, ROIs, frequency_bins)
    For the nbest_regions ROIs (where i make sure to include the important regions).
    Selected are the regions where the (Rest-MI)/(pooled std) has the max values.
    ----------
    - nbest_regions : Number of most informative regions to select.
    - important regions: Regions to be included for sure
    ----------
    Returns - selected_regions: Array
    """
    ntotROIs = wpsdMI.shape[1]
    cohen_maxs = []

    for roi in range(ntotROIs):
        wpsdMI_1ROI =wpsdMI[:,roi,:]
        wpsdRest_1ROI =wpsdRest[:,roi,:]
        # Correct order is Motor-Imagery, Rest
        # I have to pick the max of MI-Re, not the opposite
        cohen_maxs.append(cohens_d_per_columm(wpsdMI_1ROI,wpsdRest_1ROI))
    ordine_cohen = np.argsort(cohen_maxs)
    selected_regions = [i for i in important_regions]
    effect_sizes = [ cohen_maxs[_] for _ in important_regions]
    for i in range(ntotROIs):
        if len(selected_regions) >= nbest_regions:
            break
        if ordine_cohen[-i-1] not in selected_regions:
            selected_regions.append(int(ordine_cohen[-i-1]))
            effect_sizes.append(cohen_maxs[ordine_cohen[-i-1]])

    return selected_regions, effect_sizes

def global_selection(selection_method:str,DataType:str, fino_a = 15, filepath = "../Results/RegionSelection/Selected_regions"):
    """
    Once I have the selections for each subject (on all data, not dividing by folds)
    I perform the GLOBAL selection of the regions that are the best in general for all subjects

    selection_method: "MyCriteria" or "Cohen_effect_size"
    DataType: "EEG" or "MEG"
    fino_a : select among the "fino_a" best rois for each subject
    filepath: file with a col "ROIs_All_trials"

    """
    soglias_cohen = {"EEG":{10:6,15:8},"MEG":{10:7,15:10}}
    soglias_mycriteria = {"EEG":{10:6,15:8},"MEG":{10:7,15:9}}
    soglias = soglias_cohen if selection_method=="Cohen_effect_size" else soglias_mycriteria if selection_method=="MyCriteria" else None

    soglia = soglias[DataType][fino_a] # 7 per MEG,finoa10 - 10 per MEG,finoa15 -  6 per EEG,finoa10 - 8 per EEG,finoa15

    regioni_selected = pd.read_pickle(filepath+f"_{selection_method}_{DataType}_among_5_folds_training_selected_freq_8_30.pkl")["ROIs_All_trials"].to_numpy()

    regioni_selected = np.array(regioni_selected.tolist()) 

    counts, bins = np.histogram(regioni_selected[:,0:fino_a].flatten(),bins=np.arange(0,69))

    regioni_finali = [i for i in range(len(counts)) if counts[i] >=  soglia ]

    return regioni_finali

def extract_features_from_spectra_more_bands(wpsdMI, wpsdRest, selected_regions, frequiMI,
                    min_freq_frature=[4,8,12], max_freq_frature=[8,12,30], 
                    select_mean=True, select_max=True,
                    reshape_features=True):
    """
    Extract frequency-domain features from MI and Rest spectra of data.
    Features are the mean and max in a fixed window of freqs (min_freq_frature, max_freq_frature)
    For the selected ROIs. Selected previously
    ----------
    
    - wpsdxx -> welch spectra of the data: (n_trials x n_ROIs x n_freqs) # Works also with other way of estimating Power, not necessarly Welch (!?)
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
    # features_MI.shape (Trials, selected_ROIs,len(min_freq_feature*2))
    
    if reshape_features:
        features_MI = features_MI.reshape(nTrials_MI,-1)
        features_Rest = features_Rest.reshape(nTrials_Rest,-1)

    labels_MI = np.array([1 for _ in range(nTrials_MI)])
    labels_Rest = np.array([0 for _ in range(nTrials_Rest)])

    return features_MI, features_Rest, labels_MI, labels_Rest, selected_regions

def sort_rank(rankings):
    """
    Useful for representation!
    If i have a way to rank regions based on their importance among folds
    this programs sorts the regions'numbers based on who has the better rank
    (ex, region 23 is alwais in position 0, it becomes region 0 after the algorithm)

    rankings: numpy array (folds, region sorted based on their rank)
    return:
    rankings_sorted: numpy array (folds, renamed region sorted based on their rank)
    sorted_regions: correspondence between old numbers of the reion and new ones: regions sorted on mean rank among folds
    """

    unique_regions = np.unique(rankings)
    avg_positions = {}

    for region in unique_regions:
        # column index = rank position
        positions = np.where(rankings == region)[1]
        avg_positions[region] = positions.mean()

    # Sort regions by their average rank (importance)
    sorted_regions = [r for r, _ in sorted(avg_positions.items(), key=lambda x: x[1])]
    region_to_new_index = {region: i for i, region in enumerate(sorted_regions)}
    rankings_sorted = np.vectorize(region_to_new_index.get)(rankings)

    return rankings_sorted, sorted_regions

def plot_hist2d_selected_regions_folds(df_path, subject, num_best_selected=68, plot_sorted=True, figtitle="", save_path=False):
    
    
    if not plot_sorted:
        num_best_selected=68

    binx = np.arange(69)-0.001
    biny = np.arange(num_best_selected+1)-0.001

    rankings = read_ranks_subect(df_path, subject)

    n_folds = rankings.shape[0]
    # rankings shape (n_folds,68) - shape2 (68) contains the number of the regions, sorted as best region in that fold

    plt.figure(figsize=(13,3.8*num_best_selected/30))
    
    # Colors
    mic = matplotlib.colormaps.get_cmap("viridis")
    color0 = mic(0.)
    other_colors = mic(np.linspace(0.07, 1., 1001))
    colors = np.vstack([color0, other_colors])
    cmap = ListedColormap(colors)
    
    # Sort regions by their average rank among folds
    rankings_sorted, sorted_regions =  sort_rank(rankings) if plot_sorted else (rankings, None)
    rankings_sorted = rankings_sorted[:,:num_best_selected]

    # Plot
    plt.hist2d(
        rankings_sorted.flatten(),
        np.tile(np.arange(num_best_selected), (n_folds,1)).flatten(),
        bins=(binx, biny),
        cmap=cmap, vmin=0, vmax=len(rankings_sorted)  # ensures n_folds+1 discrete levels
    )

    coco = "C0"
    plt.vlines(binx,0,num_best_selected,alpha=0.75,color=coco)
    plt.hlines(biny,0,68,alpha=0.75,color=coco)

    plt.colorbar(label="how many times a region has a specific rank")
    plt.xlabel("Region")
    plt.ylabel("Rank")
    plt.title(figtitle)
    if plot_sorted:
        plt.plot([0,68],[0,68],color="red",linestyle="--",alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    if plot_sorted:
        return rankings_sorted, sorted_regions

################################## Test -  feature selection ##################################
def select_regions_MyCriteria2(wpsdMI, wpsdRest,
                    nbest_regions=8, 
                    important_regions=[4, 5, 32, 33, 44, 45, 48, 49]):
    """
    NORMALIZING with respect to the rest in the same bin where I found max(MI-Re),
    not normalizing respect to the max of rest among all the bins

    Select regions (on provided spectra)
    wpsdMI.shape = (trials, ROIs, frequency_bins)
    For the nbest_regions ROIs (where i make sure to include the important regions).
    Selected are the regions where the max(Rest-MI) / max(Rest) has the max values.
    ----------

    - nbest_regions : Number of most informative regions to select.
    - important regions: Regions to be included for sure
    
    Returns - selected_regions: Array
    """
    ######################## Search the best regions ########################
    # (regions where the the %difference at the freq with max difference is the highest)
    # I choose the nregions where I have the greatest difference between MI and Rest 
    # (but being careful to include the Interesting_regions)
    ntotROIs = wpsdMI.shape[1]
    max_diff_perc = []
    for roi in range(ntotROIs):
        wpsdMI_1ROI =wpsdMI[:,roi,:]
        wpsdRest_1ROI =wpsdRest[:,roi,:]
        wpsd_MI_mean = wpsdMI_1ROI.mean(axis=0)
        wpsd_Rest_mean = wpsdRest_1ROI.mean(axis=0)
        
        diff = wpsd_Rest_mean - wpsd_MI_mean
        idx_max = np.argmax(diff)  # indice del massimo
        max_diff_perc.append(diff[idx_max] / wpsd_Rest_mean[idx_max])

    selected_regions = [i for i in important_regions]
    for i in range(ntotROIs):
        if len(selected_regions) >= nbest_regions:
            break
        if np.argsort(max_diff_perc)[-i-1] not in selected_regions:
            selected_regions.append(int(np.argsort(max_diff_perc)[-i-1]))

    return selected_regions

# deprecated 
# but still in use (maybe)
def extract_features_more_bands(data_MI, data_Rest, sfreq, starttime, endtime, nbest_regions,
                    min_freq_frature=[4,8,12], max_freq_frature=[8,12,30], 
                    kargs_welch={"fmin":4, "fmax":30,"n_per_seg":250, "n_fft":300,"n_overlap":125},
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
    standard_dict = {"fmin":4, "fmax":30,"n_per_seg":250, "n_fft":300,"n_overlap":125, "verbose":False}
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

def compute_welch_select_regions(data_MI=None, data_Rest=None, sfreq=250, starttime=250, endtime=250, nbest_regions=8, 
                    wpsdMI=None, wpsdRest=None, frequiMI=None, select_regions=True,
                    kargs_welch={"fmin":4, "fmax":30,"n_per_seg":250, "n_fft":300,"n_overlap":125, "verbose":False},
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
        standard_dict = {"fmin":4, "fmax":30,"n_per_seg":250, "n_fft":300,"n_overlap":125, "verbose":False}
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


""" But not used anymore
    def extract_features(data_MI, data_Rest, sfreq, starttime, endtime,
                        min_freq_frature, max_freq_frature, nbest_regions, 
                        kargs_welch={"fmin":"2", "fmax":"45","n_per_seg":250, "n_fft":300,"n_overlap":125},
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