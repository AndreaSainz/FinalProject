from ct_reconstruction.models.deep_back_projection import DBP
from accelerate import Accelerator

accelerator = Accelerator()
print(accelerator.state)

# training, validation and testing paths
training_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_train'
validation_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_validation'
test_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_test'


# define parameters
n_single_BP = 16
alpha = 0.297807 #percentile 95
i_0 = 100000
sigma = 0.001
max_len_train = 100
max_len_val = 20
max_len_test = 20
seed = 29072000
debug = True
batch_size = 32
epochs = 5
learning_rate = 1e-3
scheduler = True
patience = 10


model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/dbp_16_views_training_try"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/dbp_16_views_training_try.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/dbp_16_views_training_try"

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
samples = model_dbp.results("testing", 5, figure_path)
model_dbp.report_results_images(figure_path, samples)