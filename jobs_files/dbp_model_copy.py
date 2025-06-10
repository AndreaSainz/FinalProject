from ct_reconstruction.models.deep_back_projection import DBP
from accelerate import Accelerator

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
max_len_train = 10
max_len_val = 1
max_len_test = 1
seed = 29072000
debug = True
batch_size = 5
epochs = 2
learning_rate = 1e-3
scheduler = True
patience = 10


model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/dbp_90_views_training_10"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/dbp_90_views_training_10.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/dbp_90_views_training_10"

# define model arquitecture
model_dbp = DBP(model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)

# training and validation
history = model_dbp.train(training_path, validation_path, figure_path, max_len_train, max_len_val, patience)

# saving model configuration
model_dbp.save_config()

#testing model
results = model_dbp.test(training_path, max_len_test)

#getting plots and results
model_dbp.results("both", 1, figure_path)
samples = model_dbp.results("testing", 1, figure_path)
model_dbp.evaluate_and_visualize(figure_path, samples, test_path, max_len_test,
                                num_iterations_sirt=100, num_iterations_em=100,
                                num_iterations_tv_min=100, num_iterations_nag_ls=100,
                                lamda=0.0001, only_results=False)