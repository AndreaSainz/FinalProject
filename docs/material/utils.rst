Utility Functions for CT Reconstruction
=======================================

This module provides utility functions for evaluating and visualizing CT image reconstruction performance.

Included functions:

- ``compute_psnr``: Calculates Peak Signal-to-Noise Ratio.
- ``compute_ssim``: Computes Structural Similarity Index.
- ``plot_metric``: Plots training/validation metrics over epochs.
- ``show_example``: Displays reconstructed vs ground truth images.

How to Import:
--------------
After installing the `ct_reconstruction` package, you can import the utilities as:

.. code-block:: python

    from ct_reconstruction.utils.metrics import compute_psnr, compute_ssim
    from ct_reconstruction.utils.plotting import show_example, plot_metric

API Reference:
--------------

.. automodule:: ct_reconstruction.utils.metrics
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: ct_reconstruction.utils.plotting
    :members:
    :undoc-members:
    :show-inheritance: