from collections import namedtuple
from configparser import ConfigParser

import numpy as np


# From parsing, return input with the correct type (e.g. boolean True from string 'True'
# or float 1.32 from string '1.32')
def evaluate_input_type(input):
    if np.isin(input, ["all", "max", "min"]):  # Case when the name is a built-in function
        return input

    try:
        return eval(input)
    except Exception:
        if "," in input:
            return [s.strip() for s in str(input).split(",")]
        else:
            return str(input)


class ConfigFile:
    def __init__(self, path):
        config_object = ConfigParser()
        config_object.read(path)
        print("Reading config from file {}...".format(path))
        # Get settings from the config file
        self.RUNINFO = namedtuple("RUNINFO", config_object["RUNINFO"].keys())(
            *[evaluate_input_type(v) for v in config_object["RUNINFO"].values()]
        )
        self.MODELINFO = namedtuple("MODELINFO", config_object["MODELINFO"].keys())(
            *[evaluate_input_type(v) if v != "sum" else v for v in config_object["MODELINFO"].values()]
        )
        self.RUNPARAMS = namedtuple("RUNPARAMS", config_object["RUNPARAMS"].keys())(
            *[evaluate_input_type(v) for v in config_object["RUNPARAMS"].values()]
        )
        self.INPUT_FILES = namedtuple("INPUT_FILES", config_object["INPUT_FILES"].keys())(
            *[evaluate_input_type(v) for v in config_object["INPUT_FILES"].values()]
        )
        self.OUTPUT_FILES = namedtuple("OUTPUT_FILES", config_object["OUTPUT_FILES"].keys())(
            *[evaluate_input_type(v) for v in config_object["OUTPUT_FILES"].values()]
        )
        self.COMPOUND_FEATURES = namedtuple("COMPOUND_FEATURES", config_object["COMPOUND_FEATURES"].keys())(
            *[evaluate_input_type(v) for v in config_object["COMPOUND_FEATURES"].values()]
        )
        self.PROTEIN_FEATURES = namedtuple("PROTEIN_FEATURES", config_object["PROTEIN_FEATURES"].keys())(
            *[evaluate_input_type(v) for v in config_object["PROTEIN_FEATURES"].values()]
        )
