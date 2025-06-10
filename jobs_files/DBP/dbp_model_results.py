from ct_reconstruction.models.deep_back_projection import DBP
from accelerate import Accelerator
from ct_reconstruction.utils.open_files import load_model_from_config

accelerator = Accelerator()

# training, validation and testing paths
training_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_train'
validation_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_validation'
test_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_test'


# define parameters
n_single_BP = 90
alpha = 1
i_0 = 100000
sigma = 0.001
max_len_train = 1000
max_len_val = 120
max_len_test = 120
seed = 29072000
debug = True
batch_size = 5
epochs = 25
learning_rate = 1e-3
scheduler = True
patience = 10


model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/dbp_90_views_training_1000"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/dbp_90_views_training_1000.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/dbp_90_views_training_1000"

model_dbp= load_model_from_config(model_path, True)

samples = model_dbp.results("testing", 5, figure_path)
model_dbp.report_results_images(figure_path, samples)
model_dbp.report_results_table(figure_path, test_path, max_len_test, num_iterations_sirt=100, num_iterations_em=100,
                         num_iterations_tv_min=100, num_iterations_nag_ls=100, lamda=0.0001, only_results = False)