from ct_reconstruction.models.deep_filtered_back_projection import DeepFBP
from accelerate import Accelerator
import torch
from ct_reconstruction.utils.plotting import plot_learned_filter

accelerator = Accelerator()

# Mostrar el dispositivo que está utilizando
print("Dispositivo en uso:", accelerator.device)

# Mostrar cuántas GPUs están disponibles
num_gpus = torch.cuda.device_count()
print("Número de GPUs disponibles:", num_gpus)

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
max_len_train = 10000
max_len_val = 2000
max_len_test = 2000
seed = 29072000
debug = True
batch_size = 8
epochs = 5
learning_rate = 1e-3
scheduler = True
filter_type = "Filter I"
patience = 10
model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/deepfbp_training_try"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/deepfbp_training_try.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_training_try"


# define model arquitecture
model_deepfbp = DeepFBP(model_path, filter_type, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)

# training and validation
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience) #phase 1(only filter)
epochs = 2
learning_rate = 1e-3
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience, epochs, learning_rate, phase=2)
epochs = 2
learning_rate = 1e-4
history = model_deepfbp.train_deepFBP(training_path, validation_path, figure_path, max_len_train, max_len_val, patience, epochs, learning_rate, phase=3)

# saving model configuration
model_deepfbp.save_config()

#testing model
results = model_deepfbp.test(training_path, max_len_test)

#getting plots and results
model_deepfbp.results("both", 1, figure_path)
samples = model_deepfbp.results("testing", 5, figure_path)
model_deepfbp.report_results_images(figure_path, samples)
plot_learned_filter(model_deepfbp.model.learnable_filter.weights, save_path="filterI_epoch20.png")