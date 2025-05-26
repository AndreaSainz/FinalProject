from ct_reconstruction.models.deep_filtered_back_projection import DeepFBP
from accelerate import Accelerator
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
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
max_len_train = 10
max_len_val = 5
max_len_test = 5
seed = 29072000
debug = True
batch_size = 2
epochs = 1
learning_rate = 1e-3
scheduler = True
filter_type = "Filter I"
patience = 20
model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/deepfbp_proof"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/deepfbp_proof_training.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_proof"


# define model arquitecture
model_deepfbp = DeepFBP(model_path, filter_type, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)

# training and validation
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience) #phase 1(only filter)
epochs = 1
learning_rate = 1e-3
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience, epochs, learning_rate, phase=2)
epochs = 1
learning_rate = 1e-3
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience, epochs, learning_rate, phase=3)

# saving model configuration
model_deepfbp.save_config()

#testing model
results = model_deepfbp.test(training_path, max_len_test)

# define model arquitecture
model_deepfbp = DeepFBP(model_path, filter_type, False, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)

# training and validation
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience) #phase 1(only filter)
epochs = 1
learning_rate = 1e-3
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience, epochs, learning_rate, phase=2)
epochs = 1
learning_rate = 1e-3
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience, epochs, learning_rate, phase=3)

# saving model configuration
model_deepfbp.save_config()

#testing model
results = model_deepfbp.test(training_path, max_len_test)
