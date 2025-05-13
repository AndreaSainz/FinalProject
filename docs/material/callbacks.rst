EarlyStopping Callback
=======================

Implements early stopping during training to avoid overfitting.

This class monitors validation loss and stops training when the performance does not improve
after a given number of epochs (patience). It is typically used in deep learning pipelines.

How to Import:
--------------
After installing the `ct_reconstruction` package, you can import the class as:

.. code-block:: python

    from ct_reconstruction.callbacks.early_stopping import EarlyStopping


API Reference:
--------------

.. automodule:: ct_reconstruction.callbacks.early_stopping
    :members:
    :undoc-members:
    :show-inheritance: