Utility Functions for CT Reconstruction
=======================================

This module provides utility components for evaluating, visualizing, logging, and managing models in CT image reconstruction pipelines.

Included components:

- ``compute_psnr``: Compute PSNR (Peak Signal-to-Noise Ratio) from MSE.
- ``compute_psnr_results``: Compute PSNR directly from predicted and ground truth tensors.
- ``compute_ssim``: Compute SSIM (Structural Similarity Index) between images.
- ``plot_metric``: Plot metrics (e.g. loss) over training epochs.
- ``show_example``: Visualize side-by-side comparison of reconstruction and ground truth.
- ``show_example_epoch``: Save or display image comparisons at a specific training epoch.
- ``plot_different_reconstructions``: Compare deep model output with multiple classical CT reconstructions.
- ``configure_logger``: Set up and return a logger for experiment tracking.
- ``open_config_file``: Read and parse a JSON model config file.
- ``load_model_from_config``: Instantiate and return a model using a config file.

How to Import:
--------------

.. code-block:: python

    from ct_reconstruction.utils.metrics import compute_psnr, compute_ssim
    from ct_reconstruction.utils.plotting import (
        show_example, plot_metric, show_example_epoch, plot_different_reconstructions
    )
    from ct_reconstruction.utils.loggers import configure_logger
    from ct_reconstruction.utils.open_files import load_model_from_config, open_config_file

API Reference
-------------

Metrics
-------

.. automodule:: ct_reconstruction.utils.metrics
    :members:
    :undoc-members:
    :show-inheritance:

Plotting
--------

.. automodule:: ct_reconstruction.utils.plotting
    :members:
    :undoc-members:
    :show-inheritance:

Loggers
-------

.. automodule:: ct_reconstruction.utils.loggers
    :members:
    :undoc-members:
    :show-inheritance:

Open Files
----------

.. automodule:: ct_reconstruction.utils.open_files
    :members:
    :undoc-members:
    :show-inheritance: