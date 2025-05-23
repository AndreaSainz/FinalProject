import os
import astra
import tomosipo as ts
from matplotlib import pyplot as plt
from ct_reconstruction.datasets.dataset import LoDoPaBDataset
from ct_reconstruction.models.deep_back_projection import DBP
import h5py
import torch
from accelerate import Accelerator

accelerator = Accelerator()

# training, validation and testing paths
<<<<<<< Updated upstream
training_path = '/rds/user/as3628/hpc-work/final_project_dis/as3628/data/ground_truth_train'
validation_path = '/rds/user/as3628/hpc-work/final_project_dis/as3628/data/ground_truth_validation'
test_path = '/rds/user/as3628/hpc-work/final_project_dis/as3628/data/ground_truth_test'
=======
training_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_train'
validation_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_validation'
test_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_test'
>>>>>>> Stashed changes


# define parameters
n_single_BP = 16
alpha = 1
i_0 = 100000
sigma = 0.001
max_len_train = 5
max_len_val = 5
max_len_test = 5
seed = 29072000
debug = True
batch_size = 5
epochs = 1
learning_rate = 1e-3
scheduler = True
patience = 20
<<<<<<< Updated upstream
model_path = "/rds/user/as3628/hpc-work/final_project_dis/as3628/models/dbp_first_model"
log_file = "/rds/user/as3628/hpc-work/final_project_dis/as3628/models/logs/dbp_first_model_training.log"
figure_path = "/rds/user/as3628/hpc-work/final_project_dis/as3628/models/figures/dbp_first_model"
=======
model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/dbp_first_model"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/dbp_first_model_training.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/dbp_first_model"
>>>>>>> Stashed changes


# define model arquitecture
model_dbp = DBP(model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)

# training and validation
history = model_dbp.train(training_path, validation_path, max_len_train, max_len_val, patience)

# saving model configuration
model_dbp.save_config()

#testing model
results = model_dbp.test(test_path, max_len_test)

#getting plots and results
model_dbp.results("both", 1, figure_path)
samples = model_dbp.results("testing", 15, figure_path)
model_dbp.report_results_images(figure_path, samples)
model_dbp.report_results_table(figure_path, num_iterations_sirt=200, num_iterations_em=200,
                         num_iterations_tv_min=200, num_iterations_nag_ls=200, lamda=0.0001)



