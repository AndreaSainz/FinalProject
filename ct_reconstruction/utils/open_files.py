import os
import json
from ..models.deep_back_projection import DBP
#from ..models.deep_filtered_back_projection import DeepFBP

MODEL_REGISTRY = {
    "DBP": DBP,
    #"DeepFBP": DeepFBP,
}


def open_config_file(path,debug):
    """
    Opens and prints the contents of a model configuration file.

    Args:
        path (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed configuration dictionary.
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
    Loads a model instance from a saved config JSON file.

    Args:
        config_path (str): Path prefix to the config file (without _config.json).

    Returns:
        ModelBase: Instantiated model with loaded configuration.
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

    # initilize the model
    model = ModelClass(**config)

    return model






