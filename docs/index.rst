Welcome to ct_reconstruction's documentation!
=============================================

This package provides utilities for CT image reconstruction using deep learning.
It includes datasets, model architectures, training pipelines, and evaluation tools.

.. note::
   Developed for the 
   `MPhil in Data Intensive Science <https://mphildis.bigdata.cam.ac.uk>`_ by Andrea Sainz Bear.

.. toctree::
   :maxdepth: 2
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

- **LoDoPaB Dataset support**: Easy integration with low-dose CT scan data.
- **Custom model training**: Deep backprojection architecture and flexible training loop.
- **Evaluation tools**: Compute PSNR, SSIM, and visualize results.
- **Callback system**: Early stopping and logging tools for training workflows.


Installation
------------

Ensure you have Python ≥3.11. Install the package and its dependencies with:

.. code-block:: bash

   pip install -e .


Usage
-----

A minimal example:

.. code-block:: python

   from ct_reconstruction.datasets import LoDoPaBDataset
   from ct_reconstruction.models.deep_back_projection import DBP

   dataset = LoDoPaBDataset("path/to/data")
   model = DBP()
   # Train, evaluate, or visualize...


Contributing
------------

Contributions are welcome! Fork this repository and submit a pull request.

License
-------

This project is licensed under the MIT License - see the `LICENSE` file for details.