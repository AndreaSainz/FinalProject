import os
import astra
import tomosipo as ts
from matplotlib import pyplot as plt
from ct_reconstruction.datasets.dataset import LoDoPaBDataset
from ct_reconstruction.models.deep_back_projection import DBP
import h5py
import torch


# training, validation and testing paths
training_path = '/rds/user/as3628/hpc-work/final_project_dis/data/ground_truth_train'
validation_path = '/rds/user/as3628/hpc-work/final_project_dis/data/ground_truth_validation'
test_path = '/rds/user/as3628/hpc-work/final_project_dis/data/ground_truth_test'


# define parameters
n_single_BP = 16
alpha = 0.05
i_0 = 100000
sigma = 0.1
max_len_train = 320
max_len_val = 160
max_len_test = 160
seed = 29072000
debug = True
batch_size = 32
epochs = 128
learning_rate = 1e-5
scheduler = True
patience = 20
model_path = "/rds/user/as3628/hpc-work/final_project_dis/models/dbp_first_model"
log_file = "/rds/user/as3628/hpc-work/final_project_dis/models/logs/dbp_first_model_training.log"
figure_path = "/rds/user/as3628/hpc-work/final_project_dis/models/figures/dbp_first_model"

# define model arquitecture
model_dbp = DBP(in_channels, model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, scheduler, log_file)

# training and validation
history = model_dbp.train(training_path, validation_path, max_len_train, max_len_val, patience)

# saving model configuration
model_dbp.save_config()

#testing model
results = model_dbp.test(test_path, max_len_test)

#getting plots and results
model_dbp.results("both", 1, figure_path)
amples = model_dbp.results("testing", 15, figure_path)
model_dbp.report_results_images(figure_path, samples)



