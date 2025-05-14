LoDoPaBDataset Class
=====================

Provides a custom PyTorch `Dataset` for loading and preprocessing CT images from the LoDoPaB-CT dataset.

This class loads HDF5 files, simulates realistic noisy sinograms, and generates input-output 
pairs for supervised deep learning tasks in CT reconstruction. It supports:

- **Gaussian and Poisson noise simulation** — for realistic data.
- **Single-angle backprojection generation** — useful for low-dose CT reconstruction.
- **Invalid sample filtering** — ensures only valid training data is used.

How to Import:
--------------
After installing the `ct_reconstruction` package, you can import the class as:

.. code-block:: python

    from ct_reconstruction.datasets.dataset import LoDoPaBDataset

API Reference:
--------------

.. automodule:: ct_reconstruction.datasets.dataset
    :members:
    :undoc-members:
    :show-inheritance: