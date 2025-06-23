from ct_reconstruction.models.fusion_filtered_back_projection import FusionFBP
from ct_reconstruction.utils.plotting import plot_learned_filter
from accelerate import Accelerator

accelerator = Accelerator()

# training, validation and testing paths
training_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_train'
validation_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_validation'
test_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_test'


# define parameters
n_single_BP = 90
sparse_view = False
view_angles = 90
alpha = 1
i_0 = 25000
sigma = 0.001
max_len_train = 2000
max_len_val = 240
max_len_test = 240
seed = 29072000
debug = True
batch_size = 10
epochs = 50
filter_type = "FilterII"
learning_rate = 1e-3
scheduler = True
patience = 15


model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/fusiondbp_2000_lowdose_25_FilterII"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/fusiondbp_2000_lowdose_25_FilterII.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/fusiondbp_2000_lowdose_25_FilterII"

# define model arquitecture
model_fusionfbp = FusionFBP(model_path, filter_type, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file)
plot_learned_filter(model_fusionfbp.model.learnable_filter, angle_idx=0, angle = 0, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/fusiondbp_2000_lowdose_25_FilterII_0_initial")
plot_learned_filter(model_fusionfbp.model.learnable_filter, angle_idx=499, angle = 90, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/fusiondbp_2000_lowdose_25_FilterII_90_initial")
plot_learned_filter(model_fusionfbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/fusiondbp_2000_lowdose_25_FilterII_mean_initial")
# training and validation
history = model_fusionfbp.train(training_path, validation_path, figure_path, max_len_train, max_len_val, patience)
plot_learned_filter(model_fusionfbp.model.learnable_filter, angle_idx=0, angle = 0, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/fusiondbp_2000_lowdose_25_FilterII_0_phase3")
plot_learned_filter(model_fusionfbp.model.learnable_filter, angle_idx=499, angle = 90, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/fusiondbp_2000_lowdose_25_FilterII_90_phase3")
plot_learned_filter(model_fusionfbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/fusiondbp_2000_lowdose_25_FilterII_mean_phase3")
# saving model configuration
model_fusionfbp.save_config()

#testing model
results = model_fusionfbp.test(training_path, max_len_test)

#getting plots and results
model_fusionfbp.results("both", 1, figure_path)
samples = model_fusionfbp.results("testing", 5, figure_path)
model_fusionfbp.evaluate_and_visualize(figure_path, samples, test_path, max_len_test,
                                num_iterations_sirt=100, num_iterations_em=100,
                                num_iterations_tv_min=100, num_iterations_nag_ls=100,
                                lamda=0.0001, only_results=False)