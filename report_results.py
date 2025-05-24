from ct_reconstruction.models.deep_back_projection import DBP
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
max_len_train = 1
max_len_val = 1
max_len_test = 10
seed = 29072000
debug = True
batch_size = 32
epochs = 1
learning_rate = 1e-3
scheduler = True
patience = 20


model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/report_results"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/report_results_training.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/report_results"

# define model arquitecture (just for using the function)
model_dbp = DBP(model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)

#getting report results (other CT reconstructions for the hole test set)
model_dbp.report_results_table(figure_path, test_path, max_len_test, num_iterations_sirt=200, num_iterations_em=200,
                         num_iterations_tv_min=200, num_iterations_nag_ls=200, lamda=0.0001, only_results = True)