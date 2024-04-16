import argparse
import logging

import numpy as np

from src.config import ConfigFile
from src.model import Model
from src.pdbbind_dataset import PDBBindDataset


def parse_params(config_file=None):
    try:
        if not config_file:
            parser = argparse.ArgumentParser(
                description="Parsing of config file name", formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

            parser.add_argument("-c", "--config", type=str, help="path to config file", default="configs/config.ini")
            args, unknown = parser.parse_known_args()
            if unknown != []:
                logging.warning(f"Unknown arguments {unknown}")
            config = vars(args)
            return ConfigFile(config["config"])
        else:
            return ConfigFile(config_file)
    except Exception:
        raise ImportError("Usage: pdbbind_pipeline.py [-c path_to_config.ini]")


def get_num_model_params(params, metadata):
    model = Model(
        hidden_channels=params.MODELINFO.hidden_channels,
        metadata=metadata,
        encoder=params.MODELINFO.encoder_type,
        link_type=params.MODELINFO.link_prediction_mode,
        loss_function=params.RUNPARAMS.loss_function,
        aggregation_operator=params.MODELINFO.conv_aggregation_operator,
        final_activation=params.MODELINFO.final_activation,
        num_encoding_layers=params.MODELINFO.num_encoding_layers,
        with_linears=params.MODELINFO.conv_with_linears,
        use_dropout=params.MODELINFO.use_edge_decoder_dropout,
    )
    # print number of parameters for model
    params = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(e.size()) for e in params])
    print(f"Number of model parameters:model {params:d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parsing of PDB Bind dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--pdbbind_set", type=str, help="Choice of PDB Bind dataset", default="complete")
    parser.add_argument("--config_file", type=str, help="Path to config file", default="configs/config.ini")
    parser.add_argument(
        "--split_type",
        type=str,
        help="Type of split ('core' for the 2016 CASF Core Split or 'temporal' for the 2019 Hold-out Split)",
        default="core",
    )

    args, unknown = parser.parse_known_args()
    pdbbind_set = vars(args)["pdbbind_set"]
    config_file = vars(args)["config_file"]
    split_type = vars(args)["split_type"]
    assert split_type in ["core", "temporal"], "PDBBind split type must be either 'core' or 'temporal'"

    params = parse_params(config_file=config_file)
    dataset = PDBBindDataset(params)

    get_num_model_params(params, dataset.get_metadata(use_dtis=params.MODELINFO.use_dtis))
