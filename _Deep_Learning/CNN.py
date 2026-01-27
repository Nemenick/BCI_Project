# nohup .venv/bin/python _Deep_Learning/CNN.py &> _Deep_Learning/LOGS/CNN_out_2026_01_16_10_30.txt

import time
from datetime import date
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import subprocess
import matplotlib.pyplot as plt
import subprocess
import pandas as pd 
from keras import optimizers
from keras.callbacks import EarlyStopping
import sys
from Deep_Library_BCI import MultiTaskModel

from Utils import freq_filter, split_train_val_test, windowize, save_training_results, evaluate_classification_by_region, aggregate_columns_dfs
Normalization = "RegionWise" # "RegionWise" or "TraceWise" or "mediatrace,stdregionwise"
from Utils import RegionWiseStandardizer


sys.path.append(os.path.abspath("Codes"))
from BCI_Library import read_subject
data_folder="Data/"
today = date.today()
# print(f"\n\nCurrent working directory: {os.getcwd()}") # Current working directory: /home/silvia/Documents/GitHub/BCI_Project

path = "_Deep_Learning/Models_trained/"
Script_name = "_Deep_Learning/CNN.py"
Additional_Script_name = "_Deep_Learning/Deep_Library_BCI.py"

tag = "First_Try"
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H_%M_%S")
tag = formatted_time + "_" + tag # /home/silvia/Documents/GitHub/GAN_Prova/GAN/WGAN/tag_time 
savedir = path+tag+"/"

start = 3           # seconds where to start to extract windows
sampling_hz = 250;  start = start*sampling_hz
input_shape = 128   # length of each window
shift = 62          # points to shift for next window in data
num_windows = 11    # how many windows to extract from each trial

end = start + (num_windows-1)*shift + input_shape  # seconds where to end to extract windows (1500 == 6 seconds)

epochs = 200
batch_monitor=35
latent_dim = 16
BATCH_SIZE = 256
pazienza = 7
LAST_LAYER_ACTIVATION = "sigmoid"
tanh = False


train_percentage = 0.7
validation_percentage = 0.15
test_percentage = 0.15
trp = train_percentage; vp = validation_percentage; tep = test_percentage


try:
    def get_gpu_memory_usage():
        command = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        memory_usage = subprocess.check_output(command, shell=True).decode("utf-8").strip().split("\n")
        memory_usage = [int(memory) for memory in memory_usage]
        return memory_usage
    mem_usage = get_gpu_memory_usage()
    # Allow memory growth to prevent TensorFlow from allocating all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Specify which GPU to use for this script
    # For example, to use GPU 1: tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_visible_devices(gpus[np.argmin(mem_usage)], 'GPU')
except Exception as e:
    print(f"VADO AVANTI, non scelgo device tf: Occurred {e} \n")

os.mkdir(path+tag)
# Backup scripts
with open(Script_name, "r") as f:
    script_content = f.read()
with open(savedir+'_Backup_script.txt', "w") as f:
    f.write(script_content)

with open(Additional_Script_name, "r") as fu:
    script_content = fu.read()
with open(savedir+'_Backup_script.txt', "a") as f:
    f.write("\n\n\n"+"#"*150+"\n"+"#"*45+"  Deep_Library_BCI details  "+"#"*45+"\n"+script_content)

# TODO: 
# COME NORMALIZZARE? (VAEGG esclude tutti quelli superiori a 400 microV; BrainOmni eachchannel is normalised to zero mean and unitvariance 
# at sample level)

subject = 2
DataType = "EEG"

##########################################
# Read data - filter
# TODO ATTENTION to put Rest before, then MI (for create_labels function)  
data_2_Rest = read_subject(data_folder, DataType, "Baseline",subject)
data_2_MI = read_subject(data_folder, DataType, "MI",subject)
data = np.concatenate((data_2_Rest, data_2_MI), axis=0)

data = freq_filter(data, sf=250, freqs=[4,45], type_filter="bandpass") # axis = -1 by default
data = data[:,:,start:end]        #  data.shape = (192, 68, 748)

########################################## 
# Split - Extract windows
# preserve balancing of data

start = time.perf_counter()

evaluated_by_regions_dataframes = []
# seeds_for_splits [42, 224, 647, 157, 2005]
for split_num,random_seed in enumerate([42, 224, 647, 157, 2005]):
    train_xy, val_xy, test_xy = split_train_val_test(data, train_percentage=trp, validation_percentage=vp, test_percentage=tep, seed=random_seed)

    x_train, y_train = windowize(*train_xy, win_len=input_shape, shift=shift)
    x_val, y_val     = windowize(*val_xy, win_len=input_shape, shift=shift)
    x_test, y_test   = windowize(*test_xy, win_len=input_shape, shift=shift)
    # shape: x -> (n_windows, timepoints) ; y -> (n_windows,)
    # y is ± region index (SIGN is negative for REST and positive for MI; absolute value is region index)

    print(f"\n\nx_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {x_test.shape}\n\n")

    ##################################################
    # NORMALIZATION (per region?)
    # Optimizing EEG ICA Decomposition with Machine Learning: A CNN-Based Alternative to EEGLAB for Fast and Scalable Brain Activity Analysis
    # Assessing the Role of EEG Biosignal Preprocessing to Enhance Multiscale Fuzzy Entropy in Alzheimer’s Disease Detection
    # (when applying on multiple subjects) Cross-Subject EEG-Based Emotion Recognition Through Neural Networks With Stratified Normalization 

    scaler = RegionWiseStandardizer()
    x_train = scaler.fit_transform(x_train, y_train)
    x_val = scaler.transform(x_val, y_val)
    x_test = scaler.transform(x_test, y_test)

    # reconstruction loss: MSE
    # reconstruction metric: MSE

    # classification loss: binary_crossentropy
    # classification metric: accuracy

    model = MultiTaskModel(use_decoder=False, use_classifier=True)
    optimizer = optimizers.Adam(epsilon=1e-04)
    model.compile_cases(optimizer, loss_reconstruction=None, loss_classification="binary_crossentropy")

    labels_train = (y_train > 0).astype(int)
    labels_val = (y_val > 0).astype(int)
    labels_test = (y_test > 0).astype(int)

    storia = model.fit_cases(x_train, x_val, y_train=labels_train, y_val=labels_val, epochs=epochs, batch_size=BATCH_SIZE, 
                                callbacks=EarlyStopping(monitor="val_loss", patience=pazienza,  restore_best_weights=True))
    # EarlyStopping comments:
    # val_loss in multi-output monitors the total loss (weighted);
    # patience 10 is good For Classification only
    save_training_results(model, storia, savedir+f"Split_{split_num}/")


    # Compute performances varying the region
    tmp_df = evaluate_classification_by_region(model, x_test, y_test, save_name=savedir+f"region_performance_split_{split_num}.csv", threshold=0.5)
    tmp_df["Split_seed"] = random_seed
    tmp_df["DataType"] = DataType
    tmp_df["subject"] = subject
    tmp_df["Classifier"] = "MLP"
    tmp_df["Features"] = "CNN_extracted"
    tmp_df["Comments"] = "-"
    tmp_df["Date"] = today
    tmp_df["WindowsNormalization"] = Normalization
    evaluated_by_regions_dataframes.append(tmp_df)

aggregated_df = aggregate_columns_dfs(evaluated_by_regions_dataframes)
aggregated_df.to_pickle(f"{savedir}aggregated_region_performance.pkl")

print("\n\n\nTEMPOO per 5 folds", time.perf_counter()-start, "\n\n\n")