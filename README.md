# Final Project
This is a private repository for the final project submission for the MPhil in Data Intensive Science.

The project implements a complete, modular deep learning pipeline for low-dose CT image reconstruction, built around a custom Python package: `ct_reconstruction`.

## Declarations of use of autogeneration tools 

This project made active use of **OpenAI's ChatGPT** as a development assistant throughout its creation.

### Areas Where ChatGPT Was Used

- **Docstring Generation**  
  ChatGPT was used to autogenerate consistent, professional docstrings for all files, classes, and functions.

- **Debugging Help**  
  Assisted in resolving tensor shape mismatches, data loader bugs, and training pipeline errors.

- **Plotting Improvements**  
  Helped refine plotting utilities to better represent training and test performance visually, exactly as intended.

- **Logging & Callbacks**  
  The `configure_logger` function and `EarlyStopping` class were developed with guidance from ChatGPT for clarity and maintainability.

- **DeepFBP Class Construction**  
  ChatGPT was instrumental in turning my custom filtered backprojection and backprojector scripts into a modular and learnable PyTorch class (`DeepFBP`). It suggested proper `nn.Module` subclassing, tensor-safe differentiable layers, and forward-pass logic integration.

- **Code Refactoring & Optimization**  
  Provided suggestions for making code more modular, scalable, and readable without sacrificing performance.

- **Sphinx Documentation**  
  Assisted with automating `sphinx-apidoc` integration and helped generate full example docstrings for the `DeepFBP` and `DBP` classes.

- **Example Completion**  
  Helped complete Jupyter notebook examples and training scripts by inspecting partial code.

- **README Generation**  
  This very README was drafted and refined with help from ChatGPT.

---

### Example Interaction

**Prompt:**
> Could you improve the docstring of this function that calculates PSNR and tell me what a good file-level docstring would look like?

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
For the correct operation of the coursework follow the following steps.

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

The specific commmands for the environment are:

For activating the environment:

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
```
At this point, you are prepared to execute the Jupyter notebooks with the coursework. 

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
│   └── early_stopping.py         # EarlyStopping with logging
├── datasets/
│   └── dataset.py                # LoDoPaBDataset loading, noise simulation, projection logic
├── models/
│   ├── deep_back_projection.py   # Deep BackProjection architecture
│   ├── deep_filtered_back_projection.py  # Learnable Deep FBP using tomosipo
│   └── model.py                  # Wrapper model loader for training, validation and testing
├── utils/
│   ├── loggers.py                # Logging configuration
│   ├── metrics.py                # PSNR, SSIM, MSE implementations
│   ├── plotting.py               # Training curves & comparison plots
│   └── open_files.py             # Helpers to load pretrained models with weights
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
- `observation_train.zip`
- `observation_validation.zip`
- `observation_test.zip`

### Unzip and Organize

1. Unzip all three files.
2. Create a folder named `data` in the root of the project (if it does not already exist).
3. Move the extracted folders  into the `data/` directory.

The structure should look like:

```
project_root/
├── data/
│   ├── ground_truth_train/
│   ├── ground_truth_validation/
│   └── ground_truth_test/
│   ├── observation_train/
│   ├── observation_validation/
│   └── observation_test/
```

### Data Processing

To enable faster and more consistent data loading during training and evaluation, we recommend preprocessing the dataset using two scripts: `extract_sinograms_gt.py` and `extract_sinograms_low_dose.py`.

These scripts prepare the data by pairing each ground truth image with its corresponding sinogram, and organizing the results in a more efficient format for downstream use:

- `extract_sinograms_gt.py`: Takes the ground truth images and computes the corresponding full-dose sinograms (i.e., simulated sinograms with normal radiation levels). It saves both the ground truth image and its sinogram in the output.
- `extract_sinograms_low_dose.py`: Takes the ground truth images and the low-dose observation sinograms, combining them into a single file that includes the ground truth image and the associated low-dose sinogram.

Each script should be executed **three times**, once for each of the dataset splits (`train`, `validation`, `test`). You need to specify the input directories (from `data/ground_truth_*` and `data/observation_*`) and the output directory, which should be set to `data_sino`.

After processing, the `data_sino/` directory should have the following structure:
```
project_root/
├── data_sino/
│   ├── ground_truth_train/
│   ├── ground_truth_validation/
│   ├── ground_truth_test/
│   ├── observation_train/
│   ├── observation_validation/
│   └── observation_test/
```
Each folder contains processed files in a standardized format:

- Files in the `ground_truth_*` directories contain a full-dose sinogram and the corresponding ground truth image.
- Files in the `observation_*` directories contain a low-dose sinogram and the corresponding ground truth image.

This format ensures consistent pairing and accelerates data loading in your training pipeline.

## License

This project is released under the MIT License.



## Acknowledgments

- [LoDoPaB-CT Dataset](https://www.nature.com/articles/s41597-021-00893-z)
- [tomosipo](https://github.com/ahendriksen/tomosipo)




