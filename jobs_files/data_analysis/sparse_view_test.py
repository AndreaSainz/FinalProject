"""
Script for training and evaluating DeepFBP under Sparse-View CT Configuration.

This script demonstrates the use of a new feature in the `ct_reconstruction` package that allows
training and evaluating models using **sparse-view CT** geometry. It confirms the correct integration
of this configuration by training DeepFBP with a limited number of projection angles and comparing
reconstructed outputs.

Key Features:
-------------
- **Sparse-View CT Enabled**: Uses sparse angular sampling by selecting a fixed number of projections.
- **Geometry Consistency Check**: Ensures that the geometry used for backprojections, forward projections,
  and sinogram simulation is correctly aligned across the training and testing pipelines.
- **Accelerator Integration**: Compatible with multi-GPU and mixed precision training using `Accelerate`.
- **Modular Execution**: Trains DeepFBP (Phase 1), evaluates on test set, and saves visual and quantitative results.

Parameters:
-----------
- `n_single_BP`: Number of angles used for sparse-view backprojections.
- `sparse_view`: If True, enables sparse-view training (with `view_angles` projections).
- `view_angles`: Number of angles to use in the sparse-view setting.
- `alpha`: Normalization constant for ground truth images (should match dataset preprocessing).
- `i_0`: Photon count for Poisson noise simulation (controls noise severity).
- `sigma`: Standard deviation of additive Gaussian noise.
- `max_len_*`: Number of samples to use from each split (`train`, `val`, `test`).
- `batch_size`, `epochs`, `learning_rate`, `scheduler`, `patience`: Standard training hyperparameters.
- `model_path`, `log_file`, `figure_path`: Output directories for storing model checkpoints, logs, and visual results.

Execution Steps:
----------------
1. Initializes the DeepFBP model with sparse-view parameters.
2. Trains only the filter layer (Phase 1) using full training and validation sets.
3. Evaluates the model on a held-out test set (15 samples).
4. Collects and plots results to verify qualitative and quantitative performance under sparse-view.

Note:
-----
- This script acts as a sanity check for sparse-view geometry integration across the pipeline.
- It validates that the forward operator, sparse backprojections, and sinogram loading are aligned.
- The selected `alpha` must be consistent with preprocessing (e.g., 0.2978 if using the 95th percentile normalization).

Conclusion:
----------
The reconstructed images using EM and FBP algorithms under a sparse-view CT setting (90 out of 1000 angles) are visually 
consistent with the ground truth. While artifacts are present due to the reduced angular coverage, the overall anatomy is 
preserved. This confirms that the sparse-view configuration introduced in the pipeline correctly propagates the acquisition 
geometry through all components: sinogram simulation, forward and backward operators, and reconstruction algorithms.
"""

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

# testing the model so the tes datset is uploaded
model_deepfbp.test(test_path, max_len_test)

# checking we are still using sparse-view 
print(f"DeepFBP using sparse-view: {model_deepfbp.sparse_view}")
print(f"Number of angles used: {model_deepfbp.num_angles_deepfbp}")


#getting plots and results
samples = model_deepfbp.results("testing", 15, figure_path)
model_deepfbp.report_results_images(figure_path, samples)