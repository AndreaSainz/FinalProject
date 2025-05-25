"""
Model loading utilities for CT reconstruction.

This module provides helper functions for loading models from JSON-based
configuration files. It supports automatic deserialization of registered model
classes, assignment of training accelerators, and validation of configurations.

Typical use cases include resuming training, evaluating saved models, or running
inference in production settings.

Functions:
    - open_config_file(path, debug): Loads and optionally prints model configuration.
    - load_model_from_config(config_path, debug): Instantiates a model from configuration.
"""

import os
import json
from ..models.deep_back_projection import DBP
from ..models.deep_filtered_back_projection import DeepFBP
from accelerate import Accelerator

MODEL_REGISTRY = {
    "DBP": DBP,
    "DeepFBP": DeepFBP,
}


def open_config_file(path,debug):
    """
    Opens and parses a JSON model configuration file.

    This function reads a JSON file located at `{path}_config.json` and optionally
    prints its contents if debug mode is enabled.

    Args:
        path (str): Path prefix to the config file (excluding the `_config.json` suffix).
        debug (bool): If True, prints the parsed configuration to the console.

    Returns:
        dict: Dictionary containing the model configuration.

    Raises:
        ValueError: If the configuration file does not exist.

    Example:
        >>> config = open_config_file("checkpoints/dbp", debug=True)
    """

    # check if the path exits
    if not os.path.exists(f"{path}_config.json"):
        raise ValueError(f"File not found: {path}_config.json")

    # open file with configuration
    with open(f"{path}_config.json", "r") as f:
        config = json.load(f)

    # print values for configurations if the user wants
    if debug:
        for key in config:
            print(f"{key} : {config[key]}")

    return config


def load_model_from_config(config_path, debug):
    """
    Loads a model instance from a configuration file.

    This function reconstructs a model using its saved JSON configuration.
    It dynamically selects the appropriate model class from a registry,
    resolves the accelerator setting, and instantiates the model with the
    correct arguments.

    Args:
        config_path (str): Path prefix to the JSON config file (excluding `_config.json`).
        debug (bool): If True, prints the loaded configuration to the console.

    Returns:
        ModelBase: Instantiated model object ready for training or inference.

    Raises:
        ValueError: If the model type is not registered or the accelerator type is invalid.
        RuntimeError: If CUDA is requested but not available.

    Example:
        >>> model = load_model_from_config("checkpoints/dbp", debug=True)
        >>> output = model(torch.randn(1, 10, 128, 128))
    """

    # get the configuration dictionary
    config = open_config_file(config_path, debug)

    # get model type
    model_type =  config["model_type"]

    #check that we can work with that model
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_type}' is not registered.")

    # take the correct model
    ModelClass = MODEL_REGISTRY[model_type]

    # remove unwanted entries
    config.pop("model_type", None)

    # Parse accelerator setting
    accelerator_str = config.pop("accelerator", "cuda").lower()

    if accelerator_str == "cpu":
        accelerator = Accelerator(cpu=True)
    elif accelerator_str == "cuda":
        temp_accelerator = Accelerator()
        if not temp_accelerator.device.type.startswith("cuda"):
            raise RuntimeError("Requested 'cuda' but no CUDA device is available.")
        accelerator = temp_accelerator
    else:
        raise ValueError(f"Unknown accelerator type: '{accelerator_str}'. Use 'cpu' or 'cuda'.")

    config["accelerator"] = accelerator

    # initilize the model
    model = ModelClass(**config)

    return model