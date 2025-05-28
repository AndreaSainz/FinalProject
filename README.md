# Final Project
This is a private repository for the final project submission for the MPhil in Data Intensive Science.

The project implements a complete, modular deep learning pipeline for low-dose CT image reconstruction, built around a custom Python package: `ct_reconstruction`.

## Declarations of use of autogeneration tools 

This project made active use of **OpenAI's ChatGPT** as a development assistant throughout its creation.

### Areas Where ChatGPT Was Used

- **Docstring Generation**  
  ChatGPT was used to adjust docstrings for all files, classes, and functions to the style established.

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
  Assisted with automating `sphinx-apidoc` integration and helped generate full example docstrings from my code for the `DeepFBP` and `DBP` classes.


- **README Generation**  
  This README was drafted and refined with help from ChatGPT.

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
│   ├── observation_train/
│   ├── observation_validation/
│   └── observation_test/
```

## Data Processing

To prepare the data for training and evaluation, you need to simulate both full-dose and low-dose sinograms from the ground truth CT images.

This is done using the script `sinograms_simulation.py`, which performs the following steps:

1. **Upsampling**: The ground truth images are upsampled from 362×362 to 1000×1000 pixels using bilinear interpolation. This is done to avoid the *inverse crime* during forward projection.
2. **Projection**: The upsampled images are projected using the tomosipo **fan-beam geometry** (via `ts.cone`) to compute the clean sinogram. Although the operator is called `ts.cone`, in 2D it corresponds to a fan-beam setup, as used in clinical CT.
3. **Saving**: Two outputs are generated and saved for each image:
   - A **full-dose version**, which directly uses the clean sinogram.
   - A **low-dose version**, created by simulating realistic Poisson noise with a reduced photon count (`N₀ = 4096`), followed by normalization using a maximum attenuation coefficient (`μ_max = 81.35858`, ensuring that CT images lie in the [0, 1] range), and finally applying the Beer–Lambert transform.

Each image and its corresponding sinogram (either full-dose or low-dose) is saved in a `.hdf5` file under separate output folders.

Run `sinogram_simulation.py` once for each dataset split (`train`, `validation`, `test`). The results will be stored under `data_sino/`.

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
Each HDF5 file contains:
- `data`: the original CT image (362×362)
- `sinograms`: the simulated sinogram (either full-dose or low-dose)

This format ensures consistent pairing between ground truth images and sinograms, and accelerates data loading during training.

> **Note:** The full simulation process should take under 2 hours on GPU, depending on your hardware and parallelization.

## License

This project is released under the MIT License.


## Acknowledgments

- [LoDoPaB-CT Dataset](https://www.nature.com/articles/s41597-021-00893-z)
- [tomosipo](https://github.com/ahendriksen/tomosipo)




