from ct_reconstruction.models.deep_filtered_back_projection import DeepFBP
from accelerate import Accelerator
import torch


accelerator = Accelerator()


# training, validation and testing paths
training_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_train'
validation_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_validation'
test_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_test'

# define parameters
n_single_BP = 16
sparse_view = True
view_angles = 90
alpha = 1
i_0 = 100000
sigma = 0.001
max_len_train = 1
max_len_val = 1
max_len_test = 15
seed = 29072000
debug = True
batch_size = 8
epochs = 1
learning_rate = 1e-3
scheduler = True
filter_type = "Filter I"
patience = 10
model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/sparse_view_test"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/sparse_view_test.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/sparse_view_test"


# define model arquitecture
model_deepfbp = DeepFBP(model_path, filter_type, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)

# training and validation
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience) #phase 1(only filter)


model_deepfbp.test(test_path, max_len_test)

print(f"DeepFBP using sparse-view: {model_deepfbp.sparse_view}")
print(f"Number of angles used: {model_deepfbp.num_angles_deepfbp}")

#getting plots and results
samples = model_deepfbp.results("testing", 15, figure_path)
model_deepfbp.report_results_images(figure_path, samples)