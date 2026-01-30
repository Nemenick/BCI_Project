# nohup .venv/bin/python Codes/Temp_run.py &> Codes/LOGS/Temp_run_out_2026_01_16_10_30.txt

data_folder="Data/"


from BCI_Library import read_subject, plot_brain, train_models, extract_features_more_bands, saveresults_pickle, select_rows
from BCI_Library import sort_rank, compute_welch_select_regions, extract_features_from_spectra_more_bands,compute_welch
from BCI_Library import plot_hist2d_selected_regions_folds, train_single_model, select_regions_MyCriteria, select_regions_Cohen_effect_size
from BCI_Library import global_selection, build_compiled_mlp


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
import tqdm

seed = 224

saveresults = True
filepath = "Results/BCI_Performances/BCI_Performances_region_selection_on_folds_8_30_Hz.pkl"
file_input = filepath
file_output = filepath
miff = [8,12]
maff = [12,30]
freqs_band = tuple([(i,j) for i,j in zip(miff,maff)])
Features_name = "PowerSpectra - MeanMax band"
Region_Selection= "Cohen's d effect size selection" # "-"
Comments = "MLP hidden neurons [20], epochs 20, batch size 32"
PFR = False



n_folds=5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
today = date.today()
sfreq = 250

tutte_roi = 68
verbose = False

Classifier_names, meths = ["MLP"], ["mlp"] # ["LDA", "SVC", "RandomForest"],[LDA, SVC, RandomForestClassifier]
MLP_fit_kwargs={'epochs':20, 'batch_size':32}

for DataType in ["EEG","MEG"]:
    for Classifier_name, meth in zip(Classifier_names, meths):

        Rows = pd.DataFrame({
            "ROIs": [],
            "NROIs":[],
            "DataType": [],
            "freqs_band": [],  
            "subject": [],
            "Classifier": [],
            "Accuracy": [],
            "Features": [],
            "Comments": [],
            "Region Selection": [],
            "Date": []
        })
       
        print("\n\n"+"#"*30+"\n",DataType,Classifier_name)
        for subject_index in tqdm.tqdm(range(20)):

            data_2_MI = read_subject(data_folder, DataType, "MI", subject_index, verbose=False)
            data_2_Rest = read_subject(data_folder, DataType, "Baseline", subject_index, verbose=False)

            WMI, WRe, FreqMI = compute_welch(data_2_MI, data_2_Rest, sfreq,
                                                            kargs_welch={"fmin":miff[0], "fmax":maff[-1]})
            # WMI.shape (trials, ROIs, freqs)

            labels = [1 for __ in range(len(WMI))] + [0 for __ in range(len(WRe))] 

            Feat_extraction = extract_features_from_spectra_more_bands(WMI, WRe, [ _ for _ in range(tutte_roi)], FreqMI,
                    min_freq_frature=miff, max_freq_frature=maff, 
                    select_mean=True, select_max=True,
                    reshape_features=False)


            Feat_MI, Feat_Rest, labls_MI, labls_Rest, all_regions = Feat_extraction
            #print("\n\n"+"#"*30+"featureMI.shape",Feat_MI.shape)
            # Feat_MI.shape (Trials, selected_ROIs, features for Roi)

            Feat_All = np.concatenate((Feat_MI, Feat_Rest), axis=0)
            #print("\n\n"+"#"*30+"Feat_All.shape",Feat_All.shape)
            labels_All = np.array(labels)

            for num_best_ROIs in [3,5,10,20]:#40

                fold_accuracies = []

                for fold_idx, (train_idx, val_idx) in enumerate(kf.split([0 for _ in range(len(labels))], labels)):

                    # selecting region based on training data only
                    training_index_MI = train_idx[np.where(train_idx<len(WMI))]
                    training_index_Re = train_idx[np.where(train_idx>=len(WMI))]-len(WMI)

                    regions_selected, effect_size  = select_regions_Cohen_effect_size(wpsdMI=WMI[training_index_MI], 
                                                                    wpsdRest=WRe[training_index_MI], 
                                                                    important_regions=[], nbest_regions=num_best_ROIs)

                    
                    Feat_All_regions_selected = Feat_All[:,regions_selected,:].reshape(Feat_All.shape[0],-1)
                    #print("\n\n"+"#"*30+"Feat_All_regions_selected.shape",Feat_All_regions_selected.shape)
                    #####################################################################################################################

                    if Classifier_name.lower()=="mlp":
                        meth = build_compiled_mlp(input_dim=Feat_All_regions_selected.shape[1])
                        
                    acc, cm, model = train_single_model(Feat_All_regions_selected, labels_All,
                                                        train_idx, val_idx, method=meth, seed=224, fit_kwargs=MLP_fit_kwargs)

                    fold_accuracies.append(acc)
                    if verbose:
                        print(f"Fold {fold_idx+1} accuracy: {acc:.4f}, Confusion matrix:")
                        print(cm)

                    #####################################################################################################################

                if verbose:
                    print("\n==================================\n","Mean accuracy:", np.mean(fold_accuracies))
                    print("Std accuracy:", np.std(fold_accuracies))

                new_row = {
                    "ROIs": [regions_selected],
                    "NROIs":[len(regions_selected)],
                    "DataType": [DataType],
                    "freqs_band": [freqs_band],  
                    "subject": [subject_index],
                    "Classifier": [Classifier_name], #SVC, #RandomForest, #LDA
                    "Accuracy": [fold_accuracies],
                    "Features": [Features_name],
                    "Comments": [Comments],
                    "Region Selection": [Region_Selection],
                    "Date": [today],
                    "Personalized_Freq_Range": [PFR],
                }
                Rows=pd.concat((Rows,pd.DataFrame(new_row)))
        

        if saveresults:
            saveresults_pickle(Rows, inputfile=file_input, outputfile=file_output, backup=True)