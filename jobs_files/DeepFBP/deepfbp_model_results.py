from ct_reconstruction.models.deep_filtered_back_projection import DeepFBP
from ct_reconstruction.utils.plotting import plot_learned_filter
from accelerate import Accelerator
from ct_reconstruction.utils.open_files import load_model_from_config

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
i_0 = 5000
sigma = 0.001
max_len_train = 2000
max_len_val = 240
max_len_test = 240
seed = 29072000
debug = True
batch_size = 10
epochs = 90
learning_rate = 1e-3
scheduler = True
filter_type = "Filter II"
patience = 15
model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/deepfbp_2000_lowdose_filterII"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/deepfbp_2000_lowdose_filterII.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_2000_lowdose_filterII"


model_deepfbp= load_model_from_config(model_path, True)

samples = model_deepfbp.results("testing", 5, figure_path)
model_deepfbp.evaluate_and_visualize(figure_path, samples, test_path, max_len_test,
                                num_iterations_sirt=100, num_iterations_em=100,
                                num_iterations_tv_min=100, num_iterations_nag_ls=100,
                                lamda=0.0001, only_results=False)
plot_learned_filter(model_deepfbp.model.learnable_filter, angle_idx=0, angle = 0, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_2000_lowdose_filterII_0_phase3")
plot_learned_filter(model_deepfbp.model.learnable_filter, angle_idx=44, angle = 90, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_2000_lowdose_filterII_90_phase3")
plot_learned_filter(model_deepfbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfbp_2000_lowdose_filterII_mean_phase3")
