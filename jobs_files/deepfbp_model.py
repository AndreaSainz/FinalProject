from ct_reconstruction.models.deep_filtered_back_projection import DeepFBP
from accelerate import Accelerator
import torch
from ct_reconstruction.utils.plotting import plot_learned_filter
import time

accelerator = Accelerator()


# training, validation and testing paths
training_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_train'
validation_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_validation'
test_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_test'

# define parameters
n_single_BP = 16
sparse_view = False
view_angles = 90
alpha = 1
i_0 = 100000
sigma = 0.001
max_len_train = 5
max_len_val = 2
max_len_test = 2
seed = 29072000
debug = True
batch_size = 8
epochs = 3
learning_rate = 1e-3
scheduler = True
filter_type = "Filter I"
patience = 15
model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/deepfbp_5_filterI_nuevo"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/deepfbp_5_filterI_nuevo.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_5_filterI_nuevo"


# define model arquitecture
model_deepfbp = DeepFBP(model_path, filter_type, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)
plot_learned_filter(model_deepfbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_5_filterI_nuevo_initial")
# training and validation
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience) #phase 1(only filter)
plot_learned_filter(model_deepfbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_5_filterI_nuevo_epoch3")
epochs = 1
learning_rate = 1e-3
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience, epochs, learning_rate, phase=2)
plot_learned_filter(model_deepfbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_5_filterI_nuevo_epoch4")
epochs = 1
learning_rate = 1e-4
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience, epochs, learning_rate, phase=3)


#testing model
results = model_deepfbp.test(training_path, max_len_test)

#getting plots and results
model_deepfbp.results("both", 1, figure_path)
samples = model_deepfbp.results("testing", 1, figure_path)
model_deepfbp.report_results_images(figure_path, samples)
model_deepfbp.report_results_table(figure_path, test_path, max_len_test, num_iterations_sirt=100, num_iterations_em=100,
                         num_iterations_tv_min=100, num_iterations_nag_ls=100, lamda=0.0001, only_results = False)
plot_learned_filter(model_deepfbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_5_filterI_nuevo_5")