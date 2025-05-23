import os
import astra
import tomosipo as ts
from matplotlib import pyplot as plt
from ct_reconstruction.datasets.dataset import LoDoPaBDataset
from ct_reconstruction.models.deep_filtered_back_projection import DeepFBP
import h5py
import torch
from accelerate import Accelerator

accelerator = Accelerator()

# training, validation and testing paths
training_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_train'
validation_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_validation'
test_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_test'

# define parameters
n_single_BP = 16
alpha = 1
i_0 = 100000
sigma = 0.001
max_len_train = 1200
max_len_val = 480
max_len_test = 480
seed = 29072000
debug = True
batch_size = 12
epochs = 50
learning_rate = 1e-3
scheduler = True
filter_type = "Filter I"
patience = 20
model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/deepfbp_1200_50epoch"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/deepfbp_1200_50epoch_training.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_1200_50epoch"


# define model arquitecture
model_deepfbp = DeepFBP(model_path, filter_type, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)

# training and validation
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience=10) #phase 1(only filter)
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience=10, phase=2)
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience=10, phase=3)

# saving model configuration
model_deepfbp.save_config()

#testing model
results = model_deepfbp.test(test_path, max_len_test)

#getting plots and results
model_deepfbp.results("both", 1, figure_path)
samples = model_deepfbp.results("testing", 5, figure_path)
model_deepfbp.report_results_images(figure_path, samples)
model_deepfbp.report_results_table(figure_path, num_iterations_sirt=200, num_iterations_em=200,
                         num_iterations_tv_min=200, num_iterations_nag_ls=200, lamda=0.0001)



