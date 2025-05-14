Model Architectures and Training Base
=====================================

This module includes the core model architecture and the base class for training deep learning models for CT reconstruction.

Included components:

- ``ModelBase``: A reusable base class that handles training, validation, testing, and metric logging.
- ``DBP``: Deep Back Projection model for CT image reconstruction.

How to Import:
--------------
After installing the `ct_reconstruction` package, you can import the models as:

.. code-block:: python

    from ct_reconstruction.models.model import ModelBase
    from ct_reconstruction.models.deep_back_projection import DBP

API Reference:
--------------

.. automodule:: ct_reconstruction.models.model
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: ct_reconstruction.models.deep_back_projection
    :members:
    :undoc-members:
    :show-inheritance: