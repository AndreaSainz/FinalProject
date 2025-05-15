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

- **Code Refactoring & Optimization**  
  Provided suggestions for making code more modular, scalable, and readable without sacrificing performance.

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
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m1_coursework/as3628.git
```

or 

```bash
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/assessments/m1_coursework/as3628.git
```

#### Environment
FOR MAC WITH APPLE SILICON:

CONDA_SUBDIR=osx-64 conda create -n tomosipo python=3.9
conda install astra-toolbox/label/dev::astra-toolbox
conda install aahendriksen::tomosipo
conda install pytorch
conda install h5py



To set up the environment for this project, first, ensure Python 3.9 is installed on your system. You can verify this by running 

```bash
python3.9 --version 
```
in your terminal. 

If it is not installed, download it from <https://www.python.org/> or use your system’s package manager.

Then, create a virtual environment using 

```bash
python3.9 -m venv venv 
```
and activate it by running 

```bash
source venv/bin/activate 
```
When you are done working, deactivate the environment by running 

```bash
deactivate
```
Alternatively, if you prefer to use Conda, you can create the environment with 

```bash
conda create -n my_env python=3.9
```
and activate it with  

```bash
conda activate my_env 
```
When you are done working, deactivate the environment by running 

```bash
conda deactivate
```

#### Requirements 
Once activated, update pip with 

```bash
pip install --upgrade pip 
```
and install the required dependencies listed in the requirements.txt file using 

```bash
pip install -r requirements.txt
```

At this point, you are prepared to execute the Jupyter notebooks with the coursework. 

## Documentation
In addition to the Python dependencies listed in `requirements.txt`, you need to have **Pandoc** installed for building the Sphinx documentation.

Follow the instructions below to install Pandoc based on your operating system:

- **Ubuntu/Debian**:
```bash
   sudo apt-get install pandoc
```
- **macOS (Homebrew)**:
```bash
brew install pandoc
```
- **Conda**:
```bash
conda install -c conda-forge pandoc
```
- **Windows**:
Download and install Pandoc from the official website:
<https://pandoc.org/installing.html>

To generate documentation using Sphinx run following commands from the root of the repository:

```bash
cd docs
make html
```
Once the documentation is created open the file index.html located in:

```bash
docs/_build/html
```

## Project Structure

```
.
├── ct_reconstruction/               # Custom package for reconstruction
│   ├── callbacks/                   # EarlyStopping and callback utilities
│   ├── datasets/                    # Dataset logic and sinogram simulation
│   ├── models/                      # Model architectures (e.g., DBP)
│   └── utils/                       # Logging, plotting, and metrics
│
├── docs/                            # Sphinx documentation
├── dataset_class.ipynb             # Notebook to inspect dataset logic
├── checking_last_patient.ipynb     # Notebook for final dataset check
├── README.md                        # This file
├── Instructions.md
├── pyproject.toml                   # Project configuration
├── requirements.txt                # Python dependencies
└── LICENSE
```



## Description of Custom Package: `ct_reconstruction`

The `ct_reconstruction` package is fully modular and contains:

- `datasets/`: Implements `LoDoPaBDataset`, handles loading, preprocessing, noise simulation, and sparse-view generation.
- `models/`: Includes `DBP` model class and helpers.
- `callbacks/`: Contains `EarlyStopping` logic with logging support.
- `utils/`: Provides plotting utilities, metric functions (PSNR, SSIM), and logger configuration.

All components are integrated into a clean training pipeline using PyTorch and `accelerate`.



## License

This project is released under the MIT License.



## Acknowledgments

- [LoDoPaB-CT Dataset](https://www.visnow.org/data/lodopab)
- [tomosipo](https://github.com/ahendriksen/tomosipo)
- [pytorch-msssim](https://github.com/VainF/pytorch-msssim)
- **OpenAI ChatGPT** – for assistance with documentation, code review, modularization, and design.




