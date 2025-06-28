# Final Project
This is a private repository for the final project submission for the MPhil in Data Intensive Science.

The project implements a complete, modular deep learning pipeline for low-dose CT image reconstruction, built around a custom Python package: `ct_reconstruction`.

## Declarations of use of autogeneration tools 

This project made active use of ChatGPT and DeepL Write as writing and development assistants throughout its creation.

### Areas Where ChatGPT Was Used

- **Docstring Generation**  
  ChatGPT was used to adjust docstrings for all files, classes, and functions to the style established.

- **Debugging Help**  
  Assisted in resolving tensor shape mismatches, data loader bugs, and training pipeline errors.

- **Plotting Improvements**  
  Helped refine plotting utilities to better represent training and test performance visually, exactly as intended.

- **Logging & Callbacks**  
  The `configure_logger` function and `EarlyStopping` class were developed with guidance from ChatGPT for clarity and maintainability.

- **Model Class Constructions**  
  ChatGPT was instrumental in turning my custom filtered backprojection and backprojector scripts into a modular and learnable PyTorch class (`DeepFBP` and its derivates 'DeepFusionBP, FusionFBP'). It suggested proper `nn.Module` subclassing, tensor-safe differentiable layers, and forward-pass logic integration.

- **Code Refactoring & Optimization**  
  Provided suggestions for making code more modular, scalable, and readable without sacrificing performance.

- **Sphinx Documentation**  
  Assisted with automating `sphinx-apidoc` integration and helped generate full example docstrings from my code for the `DeepFBP` and `DBP` classes.


- **README Generation**  
  This README was drafted and refined with help from ChatGPT.

### Areas Where DeepL Was Used 

Used for academic style refinement of written text:
	-	Language Polishing
Text segments (especially in the report and README) were revised using DeepL Write to improve clarity, tone, and grammatical precision in English.
	-	**Important Note**:
DeepL Write does not generate new content, it only reformulates existing text written to be more fluent and formal. It was not used to create any technical explanations or original text.

---

### Example Interaction

**Prompt:**
> Could you improve the docstring of this function following the style i already provide and making them look more professional?

**Response:**
``` python
def compute_psnr(mse, max_val=1.0):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is a logarithmic metric that compares the ratio between the maximum possible
    pixel value and the mean squared error (MSE) between a reconstructed and a reference image.

    Args:
        mse (float or torch.Tensor): Mean squared error between reconstructed and reference images.
        max_val (float, optional): Maximum possible pixel value (default: 1.0).

    Returns:
        float: PSNR value. Returns infinity if MSE is zero.
    """
```

## Installation
For the correct operation of the code follow the following steps.

#### Copy the repository
Clone the repository to your local machine using :

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/as3628.git
```

or 

```bash
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/assessments/projects/as3628.git
```

#### Environment
To set up the environment for this project, first, ensure Python 3.9 is installed on your system. You can verify this by running 

```bash
python3.9 --version 
```
in your terminal. 

If it is not installed, download it from <https://www.python.org/> or use your system’s package manager.

Then, create an environment using

```bash
python3.9 -m venv tomosipo_env
```


For activating the environment (please note that the environment must be activated for use):

```bash
source tomosipo_env/bin/activate
```
When you are done working, deactivate the environment by running 

```bash
deactivate
```

#### Requirements 
Once **activated**, install all the requirements:

```bash
pip install -r requirements.txt
pip install -e. #this is for the ct_reconstruction package made for this project)
```
At this point, you are prepared to execute the code scripts for the projects. 

If when running any of the scripts you find that the tomosipo or tomispo algorithms packages have not been properly installed, run the following commands:

```bash
pip install git+https://github.com/ahendriksen/tomosipo.git
pip install git+https://github.com/ahendriksen/ts_algorithms.git
```

## Documentation
To generate documentation using Sphinx run following commands from the root of the repository:

```bash
cd docs
make clean #this is for when you already computed the documentation
make html
```
Once the documentation is created open the file index.html located in:

```bash
docs/_build/html
```

## Package Structure: `ct_reconstruction`

```
ct_reconstruction/
├── callbacks/
│   ├── early_stopping.py                # EarlyStopping with logging
├── datasets/
│   ├── dataset.py                       # LoDoPaBDataset loading, noise simulation, projection logic
├── models/
│   ├── deep_back_projection.py          # DBP architecture
│   ├── deep_filtered_back_projection.py # Learnable DeepFBP using tomosipo
│   ├── deep_fusion_back_projection.py   # Fusion of DeepFBP and DBP 
│   ├── fusion_filtered_back_projection.py # Fusion of DeepFBP with DBP as denoiser
│   ├── model.py                         # Wrapper model loader for training, validation and testing
├── utils/
│   ├── loggers.py                       # Logging configuration
│   ├── metrics.py                       # PSNR, SSIM, MSE implementations
│   ├── open_files.py                    # Helpers to load pretrained models with weights
│   ├── plotting.py                      # Training curves, comparison plots and report plots
└── version.py
```

## Data Downloading

To run the training and evaluation pipelines, you need the **LoDoPaB-CT dataset**. Please follow these steps:

### Download from Zenodo

Dataset page: [https://zenodo.org/records/3384092](https://zenodo.org/records/3384092)

Download the following files:

- `ground_truth_train.zip`
- `ground_truth_validation.zip`
- `ground_truth_test.zip`

### Unzip and Organize

1. Unzip all three files.
2. Create a folder named `data` in the root of the project (if it does not already exist).
3. Move the extracted folders into the `data/` directory.


The structure should look like:

```
project_root/
├── data/
│   ├── ground_truth_train/
│   ├── ground_truth_validation/
│   └── ground_truth_test/
```

## Data Processing

To prepare the data for training and evaluation, **full-dose sinograms** are simulated from ground truth CT images using forward projection.

This is handled by the script [`sinogram_simulation.py`](jobs_files/data_analysis/sinogram_simulation.py), which performs the following steps:

1. **Input Loading**  
   CT slices of size **362×362** are loaded directly from preprocessed HDF5 files. No resizing or upsampling is applied.

2. **Forward Projection**  
   Each image is projected using [**tomosipo**](https://github.com/tomographic-imaging/tomosipo)’s `ts.cone` operator to simulate a **clean, noise-free sinogram** using fan-beam geometry. Although the operator is named `cone`, in 2D it corresponds to a standard clinical fan-beam setup.

3. **Output Saving**  
   For each image, a new HDF5 file is saved containing:
   - `data`: the original CT slice.
   - `sinograms`: the corresponding clean sinogram (result of forward projection).


Each image and its corresponding sinogram is saved in a `.hdf5` file under separate output folders.

Run `sinogram_simulation.py` once for each dataset split (`train`, `validation`, `test`). The resulting files will be stored under `data_sino/`, structured as follows:
```
project_root/
├── data_sino/
│   ├── ground_truth_train/
│   ├── ground_truth_validation/
│   ├── ground_truth_test/
```
Each HDF5 file contains:
- `data`: the original CT image (362×362)
- `sinograms`: the simulated sinogram 

This format ensures consistent pairing between ground truth images and sinograms, and accelerates data loading during training.

> **Note:**  
> - The fan-beam setup uses 1000 projection angles and 513 detectors, following the LoDoPaB configuration (`src_orig_dist = 575`).  
> - GPU acceleration is supported if tomosipo is properly installed with CUDA backends.  
> - The full simulation process typically completes in under 2 hours depending on hardware and parallelization.


# Model Architectures & Training Options

This repository supports four modular and configurable models for CT reconstruction, implemented as subclasses of a generic ModelBase. Each model integrates deeply with a full training pipeline including data loading, sinogram generation, physics-based operators, evaluation metrics, visualization, and reproducibility tools.


###  DeepFBP

A physics-informed deep model with a learnable frequency filter, interpolation network, and image-space denoiser.

**Main Components**:
-	Learnable filter (shared or per-angle)
-	1D angular interpolation with depthwise convolutions
-	2D CNN residual denoiser (3 blocks)
-	Differentiable fan-beam backprojection via Tomosipo

**Training Options**:
-	filter_type: "Filter I" (shared) or "Filter II" (per-angle)
-	Full-view or sparse-view sinogram input
-	Dose control: i_0 (incident intensity), sigma (Gaussian noise), alpha (scaling)
-	phase: staged training (1: filter only, 2: +interpolation, 3: full model)


### DBP

A classic deep architecture for stacked single-angle backprojections.

**Main Components**:
- 90 single-angle backprojections (Tomosipo)
- 17-layer deep CNN:
  - 1 initial conv + ReLU
  - 15 conv+BN+ReLU blocks
  - 1 final conv layer

**Training Options**:
-	Only compatible with single_bp=True (Sparse-view)
-	Adjustable number of backprojections (n_single_BP)
-	Dose control: i_0, sigma, alpha
-	Fully compatible with early stopping, LR scheduling, and example visualization


### FusionFBP

Combines the filtering and interpolation of DeepFBP with the deep denoiser of DBP.

**Main Components**:
- Same initial arquitecture as DeepFBP (learnable filter + interpolation)
- 15-layer DBP-style CNN for image denoising

**Training Options**:
- `filter_type: "Filter I" or "Filter II"
-  Full-view or sparse-view input
-  Full noise model configuration
-  phase: staged training (1: filter, 2: +interp, 3: full)

### DeepFusionBP

A hybrid architecture using stacked backprojections as image-space features.

**Main Components**:
- Learnable filtering + interpolation (like DeepFBP)
- Single-angle differentiable backprojections (Tomosipo)
- 15-layer DBP CNN operating on stacked images

**Training Options**:
- `filter_type`: `"Filter I"` (shared) or `"Filter II"`
-  Only Sparse view
-  All kind of doses by adapting `I_0` and sigma  (control noise model)
- `phase`: staged training (1: filter, 2: +interpolation, 3: full model)

###  Model Comparison

| Model          | Filter | Interpolation | Backprojections | Denoiser       | View Mode     |
|----------------|--------|---------------|-----------------|----------------|---------------|
| DeepFBP        | ✅      | ✅             | ❌              | CNN (residual) | Full / Sparse |
| DBP            | ❌      | ❌             | ✅              | Deep CNN       | Sparse        |
| FusionFBP      | ✅      | ✅             | ❌              | DBP-style CNN  | Full / Sparse |
| DeepFusionBP   | ✅      | ✅             | ✅              | DBP-style CNN  | Sparse        |


## Training Pipeline Overview

All models inherit from ModelBase, which handles:
-  Reproducible training with fixed seeds across PyTorch, NumPy, and Python
-  LoDoPaB dataset loading with automatic noise/sinogram simulation
-  Tomosipo-based volume and projection geometry setup (fan beam)
-  full-view, sparse-view, or single backprojection modes
-  Optimizer: Adam / AdamW and loss: MSELoss / L1Loss
-  Validation with PSNR / SSIM / MSE tracking
-  Automatic model checkpointing and early stopping
-  Optional learning rate scheduler (ReduceLROnPlateau)
-  Visual logging of examples every N epochs
-  Optional CLI/user prompt to confirm architecture before training

## Evaluation & Results

The ModelBase supports flexible testing and results analysis:
- Full test pipeline with .pt output files (reconstructions, sinograms, ground truths)
- Automatic inference under different angular offsets (e.g. for generalization)
- Training and test metrics saved as .json
- Plotting of training curves: loss, PSNR, SSIM
- Visualization of prediction vs ground truth images
- Integrated evaluation against:
	-	FBP
	-	SIRT
	-	EM
	-	TV-minimization
	-	NAG-LS
- PSNR / SSIM / MSE distributions saved as .csv
- Summary statistics (mean/std) and bar plots for all algorithms

## License

This project is released under the MIT License.


## Acknowledgments

- [LoDoPaB-CT Dataset](https://www.nature.com/articles/s41597-021-00893-z)
- [tomosipo](https://github.com/ahendriksen/tomosipo)
- [DeepFBP](https://ieeexplore.ieee.org/document/10411896)
- [DBP](https://arxiv.org/abs/1807.02370)