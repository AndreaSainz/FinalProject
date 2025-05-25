Welcome to ct_reconstruction's documentation!
=============================================

This package provides utilities for CT image reconstruction using deep learning.
It includes datasets, model architectures, training pipelines, and evaluation tools.

.. note::
   Developed for the 
   `MPhil in Data Intensive Science <https://www.postgraduate.study.cam.ac.uk/courses/directory/pcphmpdis>`_ by Andrea Sainz Bear.

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   material/datasets
   material/models
   material/utils
   material/callbacks

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Features
--------

- **LoDoPaB dataset integration**  
  Seamless access to low-dose CT scan data for model training and benchmarking.

- **Modular model training framework**  
  Flexible and extensible deep learning architecture built around Deep Backprojection and Deep Filtered Backprojection networks.

- **Multi-GPU and hardware-agnostic training**  
  Fully compatible with multi-GPU setups and CPU/GPU acceleration via the `accelerate` library.

- **Robust evaluation utilities**  
  Compute image quality metrics such as PSNR and SSIM for rigorous performance assessment.

- **Comprehensive visualization tools**  
  Plot training curves, compare reconstructions side-by-side, and generate figures for publications or debugging.

- **Training callbacks system**  
  Built-in support for early stopping, custom metric logging, and training hooks to facilitate efficient experimentation.

Installation
------------

Ensure you have Python ≥3.9 installed.

Follow these steps to set up the environment:

.. code-block:: bash

   # Create and activate virtual environment
   python3.9 -m venv tomosipo_env
   source tomosipo_env/bin/activate

   # Clone the repository and install in editable mode
   git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/as3628.git
   cd as3628
   pip install -e .

   # Install dependencies
   pip install -r requirements.txt

Usage
-----

The following example demonstrates the complete workflow using the `DBP` model: training, validation, testing, and result generation.

.. code-block:: python

    from ct_reconstruction.models.deep_back_projection import DBP
    from accelerate import Accelerator

    # Hardware setup
    accelerator = Accelerator()

    # Paths to data and output directories
    training_path = "/path/to/data/ground_truth_train"
    validation_path = "/path/to/data/ground_truth_validation"
    test_path = "/path/to/data/ground_truth_test"
    model_path = "/path/to/models/dbp_16_views_proof"
    log_file = "/path/to/models/logs/dbp_16_views_proof_training.log"
    figure_path = "/path/to/models/figures/dbp_16_views_proof"

    # Model hyperparameters
    model_dbp = DBP(
        model_path=model_path,
        n_single_BP=16,
        alpha=1,
        i_0=100000,
        sigma=0.001,
        batch_size=32,
        epochs=100,
        learning_rate=1e-3,
        debug=True,
        seed=29072000,
        accelerator=accelerator,
        scheduler=True,
        log_file=log_file
    )

    # Train and validate model
    history = model_dbp.train(
        training_path=training_path,
        validation_path=validation_path,
        figure_path=figure_path,
        max_len_train=1000,
        max_len_val=500,
        patience=20
    )

    # Save configuration for reproducibility
    model_dbp.save_config()

    # Evaluate on test set
    results = model_dbp.test(test_path, max_len_test=1000)

    # Generate qualitative and quantitative result visualizations
    model_dbp.results("both", 1, figure_path)
    samples = model_dbp.results("testing", 5, figure_path)
    model_dbp.report_results_images(figure_path, samples)


Contributing
------------

Contributions are welcome! Fork this repository and submit a pull request.

License
-------

This project is licensed under the MIT License - see the `LICENSE` file for details.