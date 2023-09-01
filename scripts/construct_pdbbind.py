from scripts.pda_pipeline import parse_params, load_config
from scripts.pdbbind_pipeline import construct_pdbbind_data, construct_pdbbind_core_data
import argparse

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
    params.RUNINFO = params.RUNINFO._replace(
        run_name=params.RUNINFO.run_name + "_{}_{}".format(split_type, pdbbind_set)
    )

    output_f, tensorboard = load_config(params)
    if split_type == "core":
        data, dataset = construct_pdbbind_core_data(params, output_f, pdbbind_set)
    else:
        data, dataset = construct_pdbbind_data(params, output_f, pdbbind_set)
