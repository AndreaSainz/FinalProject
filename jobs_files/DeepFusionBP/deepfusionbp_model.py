from ct_reconstruction.models.deep_fusion_back_projection import DeepFusionBP
from ct_reconstruction.utils.plotting import plot_learned_filter
from accelerate import Accelerator

accelerator = Accelerator()

# training, validation and testing paths
training_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_train'
validation_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_validation'
test_path = '/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_test'


# define parameters
view_angles = 90
alpha = 1
i_0 = 100000
sigma = 0.001
max_len_train = 10
max_len_val = 2
max_len_test = 2
seed = 29072000
debug = True
batch_size = 5
epochs = 2
learning_rate = 1e-3
filter_type = "FilterI"
scheduler = True
patience = 10


model_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/deepfusionbp_try"
log_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/logs/deepfusionbp_try.log"
figure_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfusionbp_try"

# define model arquitecture
model_deepfusionbp= DeepFusionBP(model_path, filter_type, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed,accelerator, scheduler, log_file)
plot_learned_filter(model_deepfusionbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfusionbp_try_initial")

# training and validation
history = model_deepfusionbp.train(training_path, validation_path, figure_path, max_len_train, max_len_val, patience)
plot_learned_filter(model_deepfusionbp.model.learnable_filter, save_path="/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/deepfusionbp_try_phase3")

# saving model configuration
model_deepfusionbp.save_config()

#testing model
results = model_deepfusionbp.test(training_path, max_len_test)

#getting plots and results
model_deepfusionbp.results("both", 1, figure_path)
samples = model_deepfusionbp.results("testing", 1, figure_path)
model_deepfusionbp.evaluate_and_visualize(figure_path, samples, test_path, max_len_test,
                                num_iterations_sirt=100, num_iterations_em=100,
                                num_iterations_tv_min=100, num_iterations_nag_ls=100,
                                lamda=0.0001, only_results=False)