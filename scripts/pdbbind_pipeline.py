from datetime import datetime
import json as json
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, argparse

import torch.utils.data

import torch
from src.config import ConfigFile

from src.model import *
from src.data import *
from src.pdbbind_dataset import construct_pdbbind_data, construct_pdbbind_core_data

from torch.utils.tensorboard import SummaryWriter


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
    except:
        raise ImportError("Usage: pdbbind_pipeline.py [-c path_to_config.ini]")


def load_config(params, data_log=None, run_name=None):
    # basic logging config
    # params.base_name = os.path.splitext(os.path.split(config['config'])[1])[-2]
    if params.RUNINFO.run_name:
        params.base_name = params.RUNINFO.run_name
    else:
        params.base_name = "default"

    if not data_log:
        logging.basicConfig(filename=os.path.join("outputs", "data_" + params.base_name + ".log"), level=logging.DEBUG)
    elif data_log != sys.stdout:  # Use custom data log file - to not erase previous infos
        logging.basicConfig(filename=os.path.join("outputs", data_log), level=logging.DEBUG)

    if run_name == sys.stdout:
        output_f = sys.stdout
        tensorboard = SummaryWriter()
    elif run_name:  # Overwrite run name indicated in config file
        output_f = open(os.path.join("outputs", "{}.log".format(run_name)), mode="a+")
        tensorboard = SummaryWriter(log_dir=os.path.join("runs", "{}".format(run_name)))
    else:  # Use custom learning log file - to not erase previous infos
        learning_log = params.base_name + ".log"
        output_f = open(os.path.join("outputs", learning_log), mode="a+")
        tensorboard = SummaryWriter(log_dir=os.path.join("runs", params.base_name))

    print(f"\n\n============ RUN - {datetime.now()} ===================", file=output_f)

    return output_f, tensorboard


def init_model(params, data, dataset):
    model = Model(
        hidden_channels=params.MODELINFO.hidden_channels,
        data=data,
        metadata=dataset.get_metadata(use_dtis=params.MODELINFO.use_dtis),
        encoder=params.MODELINFO.encoder_type,
        link_type=params.MODELINFO.link_prediction_mode,
        loss_function=params.RUNPARAMS.loss_function,
        aggregation_operator=params.MODELINFO.conv_aggregation_operator,
        final_activation=params.MODELINFO.final_activation,
        num_encoding_layers=params.MODELINFO.num_encoding_layers,
        with_linears=params.MODELINFO.conv_with_linears,
        use_dropout=params.MODELINFO.use_edge_decoder_dropout,
    )

    # model_fp16 = torch.quantization.quantize_dynamic(
    # 	model,  # the original model
    # 	{torch.nn.Linear},  # a set of layers to dynamically quantize
    # 	dtype=torch.float16)  # the target dtype for quantized weights
    return model


def run_pipeline(params, data, model, output_f, tensorboard):
    device = torch.device("cuda" if torch.cuda.is_available() and not params.RUNINFO.force_cpu else "cpu")

    model = model.to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        data = data.to(device)
        model(data.x_dict, data.edge_index_dict, model.inspect(data, mode="test"), data.edge_attr_dict)
        data = data.to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=params.RUNPARAMS.learning_rate)

    # Create batches
    if isinstance(params.RUNPARAMS.batch_size, int) and params.RUNPARAMS.batch_size > 0:
        batch_size = params.RUNPARAMS.batch_size
    else:
        batch_size = data[params.MODELINFO.link_prediction_mode].y.size(0)
    batches = GraphBatchIterator(
        data,
        edge_type=params.MODELINFO.link_prediction_mode,
        batch_size=batch_size,
        use_dtis=params.MODELINFO.use_dtis,
    )

    best_iter = (-1, None, np.inf, np.inf, np.inf, np.inf, None)
    val_best_iter = (-1, None, np.inf, np.inf, np.inf, np.inf, np.inf, None)
    all_losses = np.empty((params.RUNPARAMS.epochs, 5))
    for epoch in range(params.RUNPARAMS.epochs):
        # print(f'\nEpoch {epoch}...')
        losses = torch.ones(batches.nb_batches)

        for idx, batch in enumerate(batches):
            batch = batch.to(device)
            # print(f'Before train: allocated {round(torch.cuda.memory_allocated(0)/1024**3,6)}GB')
            losses[idx] = model.train_graph(batch, optimizer, penalize_fn=params.RUNPARAMS.penalize_false_negative)
            # print(f'After train: allocated {round(torch.cuda.memory_allocated(0)/1024**3,6)}GB')
            batch = batch.to("cpu")
        # Manually clean memory
        torch.cuda.empty_cache()

        # Final pred, after all the batches has been runned
        # print(f'Before test: allocated {round(torch.cuda.memory_allocated(0)/1024**3,6)}GB')
        data = data.to(device)

        # Record val performances
        val_pred, val_loss = model.test_predict_graph(data, penalize_fn=False, set="val")

        # Record test performances
        test_pred, test_loss = model.test_predict_graph(data, penalize_fn=False, set="test")
        rmse_value = model.test_rmse(data, set="test")
        rbo_value = 1 - model.test_rbo(
            data, set="test"
        )  # Test RBO give RBO loss, here we want to highlight RBO itself (1 being perfect)
        pearson_coeff = model.test_pearson(data, set="test")  # From -1 to 1, 0 meaning no correlation
        r2_score = model.test_r2(data, device, set="test")
        train_rmse = model.test_rmse(data, set="train")
        loss = losses.mean()

        # Manually clean memory
        data = data.to("cpu")
        torch.cuda.empty_cache()
        # print(f'After test: allocated {round(torch.cuda.memory_allocated(0)/1024**3,6)}GB')

        if val_loss < val_best_iter[2]:
            val_best_iter = (
                epoch + 1,
                val_pred,
                val_loss,
                rmse_value,
                rbo_value,
                pearson_coeff,
                r2_score,
                copy.deepcopy(model),
            )
        if test_loss < best_iter[2]:
            best_iter = (epoch + 1, test_pred, rmse_value, rbo_value, pearson_coeff, r2_score, copy.deepcopy(model))
        all_losses[epoch] = [train_rmse, rmse_value, rbo_value, pearson_coeff, r2_score]
        print(
            f"Epoch: {epoch+1:03d}, Train RMSE: {train_rmse:.4f}, Test RMSE: {rmse_value:.4f}, RBO: {rbo_value:.4f}, Pearson Coeff (Rp): {pearson_coeff:.4f}, R2: {r2_score:.4f}",
            file=output_f,
        )

        # Record metrics on TensorBoard
        tensorboard.add_scalar("Train RMSE", loss, epoch)
        tensorboard.add_scalar("RMSE", test_loss, epoch)
        tensorboard.add_scalar("RBO", rbo_value, epoch)
        tensorboard.add_scalar("Pearson Coeff (Rp)", pearson_coeff, epoch)
        tensorboard.add_scalar("R²", r2_score, epoch)
        tensorboard.flush()
        output_f.flush()

        if train_rmse is np.nan and rmse_value is np.nan:
            # Issue to be resolved - early stopping so far
            logging.warning("Both train and test RMSE are NaN - gradient overflowed, stopping training")
            break

    preds = best_iter[1].flatten()
    targets = model.ground_truth(data, mode="test").float().flatten()
    print(
        f"Mean absolute difference between targets and predictions at the end: ",
        (targets - preds).abs().mean().item(),
        file=output_f,
    )

    print(
        f"Performances at best: Epoch {best_iter[0]:d} - Test RMSE = {best_iter[2]:.4f}, RBO = {best_iter[3]:.4f}, Pearson Coeff (Rp) = {best_iter[4]:.4f}, R² = {best_iter[5]:.4f}",
        file=output_f,
    )
    print(
        f"Performances at best val: Epoch {val_best_iter[0]:d} (Val RMSE = {val_best_iter[2]:.4f}) - Test RMSE = {val_best_iter[3]:.4f}, RBO = {val_best_iter[4]:.4f}, Pearson Coeff (Rp) = {val_best_iter[5]:.4f}, R² = {val_best_iter[6]:.4f}",
        file=output_f,
    )

    output_f.flush()
    # output_f.close()
    model = best_iter[6]  # Return best model trained so far
    return model, best_iter, all_losses


def plot_results(params, model, dataset, data, best_iter, all_losses, output_f):
    preds = best_iter[1].flatten()
    targets = model.ground_truth(data, mode="test").float().flatten()

    # Export predictions in a CSV file
    if model.link_from == "target" and model.link_to == "compound":
        protein_nodes = model.inspect(data, mode="test")[0, :]
        compound_nodes = model.inspect(data, mode="test")[1, :]
    elif model.link_from == "compound" and model.link_to == "target":
        compound_nodes = model.inspect(data, mode="test")[0, :]
        protein_nodes = model.inspect(data, mode="test")[1, :]
    compound_nodes = np.array(data["compound"].names)[compound_nodes.tolist()]
    protein_nodes = np.array(data["target"].names)[protein_nodes.tolist()]
    if output_f != sys.stdout:
        pd.DataFrame(
            {
                "Predictions": preds.cpu(),
                "Ground Truths": targets.cpu(),
                "Compound": compound_nodes,
                "Target": protein_nodes,
            }
        ).to_csv(output_f.name.replace(".log", ".csv"))

    # Save model
    torch.save(model.state_dict(), os.path.join("outputs", "model_" + params.base_name + ".pt"))

    # Plot loss and metrics curve
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax1.plot(np.arange(params.RUNPARAMS.epochs), all_losses[:, 0], "b-", label="Train RMSE Loss")
    ax1.plot(np.arange(params.RUNPARAMS.epochs), all_losses[:, 1], "r-", label="Test RMSE Loss")
    # ax2.plot(np.arange(params.RUNPARAMS.epochs), all_losses[:,2], 'g-', label='RBO Score')
    ax2.plot(np.arange(params.RUNPARAMS.epochs), all_losses[:, 4], "g-", label="R² Score")
    ax2.plot(np.arange(params.RUNPARAMS.epochs), all_losses[:, 3], "y-", label="Pearson Coeff")

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Root Mean Square Error", color="b")
    if max(all_losses[10:, 1:2]) < 5:
        ax1.set_ylim([-0.1, 5.1])
    else:
        ax1.set_ylim([-0.1, max(all_losses[10:, 1:2]) + 0.5])
    ax1.legend(loc="upper left")
    ax2.set_ylabel("Rank-Biased Overlap / Pearson Correlation Coefficient", color="g")
    ax2.set_ylim([-0.01, 1.01])
    ax2.legend(loc="upper right")

    np.savetxt(os.path.join("outputs", "losses_" + params.base_name + ".csv"), all_losses, delimiter=",")
    plt.savefig(os.path.join("outputs", "learning-curve_" + params.base_name + ".png"))


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

    model = init_model(params, data, dataset)
    model, best_iter, all_losses = run_pipeline(params, data, model, output_f, tensorboard)
    plot_results(params, model, dataset, data, best_iter, all_losses, output_f)
