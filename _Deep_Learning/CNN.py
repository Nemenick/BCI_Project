# nohup .venv/bin/python _Deep_Learning/CNN.py &> _Deep_Learning/LOGS/CNN_out_2026_01_16_10_30.txt

import keras
import os
from datetime import datetime
from keras import layers
import numpy as np
import tensorflow as tf
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd 
import csv
import sys
sys.path.append(os.path.abspath("Codes"))
from BCI_Library import plot_brain

# print(f"\n\nCurrent working directory: {os.getcwd()}") # Current working directory: /home/silvia/Documents/GitHub/BCI_Project

path = "_Deep_Learning/Models_trained"
script_name = "_Deep_Learning/CNN.py"
# UTILS_NAME = "/home/messuti/GitHubFolders/GAN_GAIAS2/GAN/_GAN_utils.py"

tag = "First_Try"
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H_%M_%S")
tag = formatted_time + "_" + tag # /home/silvia/Documents/GitHub/GAN_Prova/GAN/WGAN/tag_time 


# epochs = 100*40
# critic_extra_steps=4
# batch_monitor=25     # TODO 35
# latent_dim = 64
# BATCH_SIZE = 512
# LAST_LAYER_ACTIVATION = "sigmoid"
# tanh = False
# # BATCH_SIZE = 512 fin qui tutto bene

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
# Backup script
with open(script_name, "r") as f:
    script_content = f.read()
with open(path+tag+'/_Backup_script.txt', "w") as f:
    f.write(script_content)

# with open(UTILS_NAME, "r") as fu:
#     script_content = fu.read()
# with open(path+tag+'/_Backup_script.txt', "a") as f:
#     f.write("\n\n\n"+"#"*45+"GAN_utils_Details"+"#"*45+"\n"+script_content)
