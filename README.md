# M1 coursework
This is a private repository for uploading Machine Learning coursework for the Mphil in Data Intensive Science.

## Declarations of use of autogeneration tools 

In the course of developing this project, I utilised two autogenerative tools, namely ChatGPT and DeepL Write. 

ChatGPT was utilised to optimise the visual representation of graphs, facilitating alterations in colour or image distribution. Furthermore, it was employed to generate the plots derived from the Optuna study.
Additionally, it was employed to resolve minor programming issues, such as debugging errors. Moreover, it enabled the implementation of particular functions, such as TPESampler for the reproducibility of optuna studies. 

Furthermore, DeepL Write was utilised to paraphrase the text, incorporating a more formal and academic style into the code comments and report. It should be noted that DeepL Write is not a tool for the generation of text; rather, it is designed for the improvement of existing texts. 

The collective application of these tools resulted in an improvement in the clarity, functionality, and presentation of the project.

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

## Structure of the coursework

This work is comprised of six discrete files, each of which can be executed independently.

The files designated as "samples.py" and "test_samples.ipynb" correspond to the initial activity of the work, designated as "Task 1." The initial file contains the functions that generate the samples, while the subsequent file comprises an examination of the generated samples to corroborate their operational efficacy.

The file entitled "neural_network.ipynb" contains the code for task 2 of the project. 

The file entitled "sklearn_models.ipynb" contains the code for task 3 of the project. 

The file entitled "weak_linear_classifiers.ipynb" contains the code for task 4 of the project. 

The file t-SNE.ipynb contains the code for task 5 of the project. 

The folder entitled "Models" contains the saved models from the executions. The models have been loaded directly into the notebooks, thus obviating the need to re-run the Optuna studies. In the event that these studies are to be conducted and the models obtained from them are to be employed in the cells, the requisite modifications are indicated.

The majority of the results presented in this study are reproducible if the notebook is executed on the same computer on multiple occasions (it should be noted that slight discrepancies may occur when the computer is changed, due to the influence of CPU or GPU operations). This applies equally to the results derived from the Optuna studies. Nevertheless, there are instances where the results are not fully reproducible. This phenomenon is particularly evident in the case of perplexity optimisation in the t-SNE algorithm, where the order of execution is not reproducible due to the internal parallelisation of the function. Although the discrepancy in results is slight, it is not entirely predictable.
