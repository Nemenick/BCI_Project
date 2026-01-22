# nohup .venv/bin/python _Deep_Learning/CNN.py &> _Deep_Learning/LOGS/CNN_out_2026_01_16_10_30.txt

import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd 
import sys
from Deep_Library_BCI import MultiTaskModel
sys.path.append(os.path.abspath("Codes"))

from Utils import freq_filter, split_train_val_test, windowize

from BCI_Library import read_subject
data_folder="Data/"

# print(f"\n\nCurrent working directory: {os.getcwd()}") # Current working directory: /home/silvia/Documents/GitHub/BCI_Project

path = "_Deep_Learning/Models_trained/"
Script_name = "_Deep_Learning/CNN.py"
Additional_Script_name = "_Deep_Learning/Deep_Library_BCI.py"

tag = "First_Try"
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H_%M_%S")
tag = formatted_time + "_" + tag # /home/silvia/Documents/GitHub/GAN_Prova/GAN/WGAN/tag_time 


start = 3           # seconds where to start to extract windows
sampling_hz = 250;  start = start*sampling_hz
input_shape = 128   # length of each window
shift = 62          # points to shift for next window in data
num_windows = 11    # how many windows to extract from each trial

end = start + (num_windows-1)*shift + input_shape  # seconds where to end to extract windows (1500 == 6 seconds)

epochs = 200
batch_monitor=35
latent_dim = 16
BATCH_SIZE = 512
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
with open(path+tag+'/_Backup_script.txt', "w") as f:
    f.write(script_content)

with open(Additional_Script_name, "r") as fu:
    script_content = fu.read()
with open(path+tag+'/_Backup_script.txt', "a") as f:
    f.write("\n\n\n"+"#"*150+"\n"+"#"*45+"  Deep_Library_BCI details  "+"#"*45+"\n"+script_content)

# TODO: 
# COME NORMALIZZARE? (VAEGG esclude tutti quelli superiori a 400 microV; BrainOmni eachchannel is normalised to zero mean and unitvariance 
# at sample level)

subject = 2
DataType = "EEG"

data_2_Rest = read_subject(data_folder, DataType, "Baseline",subject)
data_2_MI = read_subject(data_folder, DataType, "MI",subject)
data = np.concatenate((data_2_Rest, data_2_MI), axis=0)

data = freq_filter(data, sf=250, freqs=[4,45], type_filter="bandpass") # axis = -1 by default
data = data[:,:,start:end]        #  data.shape = (192, 68, 748)

train, val, test = split_train_val_test(data, train_percentage=trp, validation_percentage=vp, test_percentage=tep)

x_train, y_train = windowize(*train, win_len=input_shape, shift=shift)
x_val, y_val = windowize(*val, win_len=input_shape, shift=shift)
x_test, y_test = windowize(*test, win_len=input_shape, shift=shift)

print(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {x_test.shape}")



##################################################
# TODO Normalization (per region)
##################################################


