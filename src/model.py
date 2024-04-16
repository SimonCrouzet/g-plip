from copy import Error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.nn import GATv2Conv, Linear, SAGEConv, to_hetero
from torchmetrics import R2Score

from src.data import get_missing_links, get_singular_edge_indices
from src.loss import (
    RBOLoss,
    pearson_correlation_coefficient,
    penalize_false_negative,
    weighted_mean_absolute_error_loss,
    weighted_mse_loss,
    weighted_rmse,
)
from src.utils import correct_negative_edges, get_sample_weights


class SAGEEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, final_activation, with_linears, in_channels=-1):
        super().__init__()
        self.final_activation = final_activation

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv((-1, -1), hidden_channels, project=with_linears))

        self.lastconv = SAGEConv((-1, -1), out_channels, project=with_linears)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        x = self.lastconv(x, edge_index)

        if self.final_activation:
            x = x.relu()

        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, final_activation, with_linears):
        super().__init__()
        self.with_linears = with_linears
        self.final_activation = final_activation

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv((-1, -1), hidden_channels, edge_dim=1, add_self_loops=False))

        self.lastconv = GATv2Conv((-1, -1), out_channels, edge_dim=1, add_self_loops=False)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = x.relu()
        x = self.lastconv(x, edge_index, edge_attr)

        if self.final_activation:
            x = x.relu()
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, link_type, final_activation, use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        if use_batchnorm:
            self.batchnorm1 = BatchNorm1d(hidden_channels)
        if use_dropout:
            self.dropout1 = Dropout(0.2)
        self.lin2 = Linear(hidden_channels, 1)

        self.link_from = link_type[0]
        self.link_to = link_type[1]
        self.final_activation = final_activation

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[self.link_from][row], z_dict[self.link_to][col]], dim=-1)

        z = self.lin1(z)
        if self.use_batchnorm:
            z = self.batchnorm1(z)
        z = z.relu()
        if self.use_dropout:
            z = self.dropout1(z)
        z = self.lin2(z).relu()
        return z


class Model(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        metadata,
        link_type,
        loss_function="cross_entropy",
        encoder="sage",
        num_encoding_layers=2,
        aggregation_operator="mean",
        final_activation=True,
        with_linears=False,
        use_dropout=False,
    ):
        super().__init__()
        self.encoder_type = encoder
        if encoder == "sage":
            self.encoder = SAGEEncoder(
                hidden_channels,
                hidden_channels,
                num_layers=num_encoding_layers,
                final_activation=final_activation,
                with_linears=with_linears,
            )
            self.encoder = to_hetero(self.encoder, metadata, aggr=aggregation_operator)
        elif encoder == "gat":
            self.encoder = GATEncoder(
                hidden_channels,
                hidden_channels,
                num_layers=num_encoding_layers,
                final_activation=final_activation,
                with_linears=with_linears,
            )
            self.encoder = to_hetero(self.encoder, metadata, aggr=aggregation_operator)
        else:
            raise Error("Encoder type {} not implemented!".format(encoder))
        self.decoder = EdgeDecoder(
            hidden_channels, link_type=link_type, final_activation=final_activation, use_dropout=use_dropout
        )
        self.loss_function = loss_function
        self.link_from = link_type[0]
        self.link_to = link_type[1]

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_attr_dict):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        return self.decoder(z_dict, edge_label_index)

    def inspect(self, data, mode="all"):
        edge_type = (self.link_from, self.link_to)
        if mode == "train":
            return data[edge_type].y_edge_index[:, data[edge_type].train_idx]
        elif mode == "val":
            return data[edge_type].y_edge_index[:, data[edge_type].val_idx]
        elif mode == "test":
            return data[edge_type].y_edge_index[:, data[edge_type].test_idx]
        elif mode == "all":
            return data[edge_type].y_edge_index
        elif mode == "singular":
            return data[edge_type].y_edge_index[:, get_singular_edge_indices(data, edge_type)]
        elif mode == "missing":
            return get_missing_links(data, edge_type)
        else:
            raise ValueError("GNN inspect(): mode can only be 'train', 'val', 'test' or 'all'")

    def ground_truth(self, data, mode="all"):
        edge_type = (self.link_from, self.link_to)
        if mode == "train":
            return data[edge_type].y[data[edge_type].train_idx]
        elif mode == "val":
            return data[edge_type].y[data[edge_type].val_idx]
        elif mode == "test":
            return data[edge_type].y[data[edge_type].test_idx]
        elif mode == "all":
            return data[edge_type].y
        else:
            raise ValueError("GNN ground_truth(): mode can only be 'train', 'val', 'test' or 'all'")

    def inspect_node(self, data, node_idx):
        node_gt = {}
        for edge_from, edge_to, y in zip(self.inspect(data)[0], self.inspect(data)[1], self.ground_truth(data)):
            if edge_from == node_idx:
                node_gt[edge_to] = y
        return list(dict(sorted(node_gt.items(), key=lambda item: item[1], reverse=True)).keys())

    def evaluate_nodes(self, x_dict, edge_index_dict, edge_label_index, edge_attr_dict, targets):
        # Compute output and give ground truths node by node - for ranking purposes
        predictions = self.forward(x_dict, edge_index_dict, edge_label_index, edge_attr_dict)
        outputs, ground_truths = [], []
        for node_idx in edge_label_index[0].unique():
            sel = torch.where(edge_label_index[0] == node_idx)
            outputs.append(predictions[sel[0]])
            ground_truths.append(targets[sel[0]])
        return outputs, ground_truths

    def calculate_loss(self, pred, target, weights=None, penalize_fn=False, reduction="none"):
        if weights is None:
            if self.loss_function == "cross_entropy":
                loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
            elif self.loss_function == "mean_square_error":
                loss = weighted_mse_loss(pred, target)
            elif self.loss_function == "mean_absolute_error":
                loss = weighted_mean_absolute_error_loss(pred, target)
            elif self.loss_function == "root_mean_square_error":
                loss = weighted_rmse(pred, target)
            elif self.loss_function == "pearson_correlation_coefficient":
                loss = pearson_correlation_coefficient(pred.flatten(), target.flatten())
            else:
                raise Error("Model: loss function {} not implemented".format(self.loss_function))
        else:
            weights = get_sample_weights(target, weights)
            target, indices = correct_negative_edges(target)
            if self.loss_function == "cross_entropy":
                loss = F.binary_cross_entropy_with_logits(
                    pred, target.float(), weight=torch.Tensor(weights), reduction="none"
                )
            elif self.loss_function == "mean_square_error":
                loss = weighted_mse_loss(pred, target, weight=weights)
            elif self.loss_function == "mean_absolute_error":
                loss = weighted_mean_absolute_error_loss(pred, target, weight=weights)
            elif self.loss_function == "root_mean_square_error":
                loss = weighted_rmse(pred, target, weight=weights)
            else:
                raise Error("Model: loss function {} not implemented".format(self.loss_function))

        if penalize_fn:
            loss = penalize_false_negative(loss, pred, target)

        if self.loss_function == "pearson_correlation_coefficient":
            return torch.sub(1, loss)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            pass
        else:
            raise Error("Model: Reduction should be set to 'mean', 'sum' or 'none'")

        return loss

    def train_graph(self, data, optimizer, weights=None, penalize_fn=False, set="train"):
        self.train()
        optimizer.zero_grad()
        pred = self(data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict)
        target = self.ground_truth(data, mode=set)  # .float()
        loss = self.calculate_loss(pred, target, weights=weights, penalize_fn=penalize_fn, reduction="mean")
        loss.backward()
        # Gradient clipping - it tends to vanish otherwise
        nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
        optimizer.step()
        return loss

    def train_deterministic_graph(self, data, optimizer, activation_level, set="train"):
        self.train()
        optimizer.zero_grad()
        pred = self(data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict)
        target = self.ground_truth(data, mode=set)
        det_target = (target >= activation_level).float()
        activated_pred = torch.sigmoid(torch.sub(pred, activation_level))
        # loss = torch.binary_cross_entropy_with_logits(activated_pred, target).mean()
        criterion = torch.nn.BCELoss()
        loss = criterion(activated_pred, det_target)
        loss.backward()
        # Gradient clipping - it tends to vanish otherwise
        nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
        optimizer.step()
        return loss

    @torch.no_grad()  # torch.no_grad() means that you are not computing gradients (saves computations)
    def test_graph(self, data, weights=None, penalize_fn=False, set="test"):
        self.eval()
        pred = self(data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict)
        target = self.ground_truth(data, mode=set)  # .float()
        loss = self.calculate_loss(pred, target, weights=weights, penalize_fn=penalize_fn, reduction="mean")
        return loss

    @torch.no_grad()
    def predict_graph(self, data, set="test"):
        self.eval()
        return self(data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict)

    @torch.no_grad()
    def test_predict_graph(self, data, weights=None, penalize_fn=False, set="test"):
        self.eval()
        pred = self(data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict)
        target = self.ground_truth(data, mode=set)  # .float()
        loss = self.calculate_loss(pred, target, weights=weights, penalize_fn=penalize_fn, reduction="mean")
        return pred, loss

    @torch.no_grad()
    def test_rbo(self, data, set="test"):
        self.eval()
        target = self.ground_truth(data, mode=set)  # .float()
        predictions, ground_truths = self.evaluate_nodes(
            data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict, target
        )
        criterion = RBOLoss(reduction="mean")
        loss = criterion(predictions, ground_truths)
        return loss.item()

    @torch.no_grad()
    def test_pearson(self, data, set="test"):
        self.eval()
        pred = self(data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict)
        target = self.ground_truth(data, mode=set)  # .float()
        # pearson = PearsonCorrCoef()
        # loss = pearson(pred, target)
        loss = pearson_correlation_coefficient(pred.flatten(), target.flatten())
        return loss.item()

    @torch.no_grad()
    def test_r2(self, data, device, set="test"):
        self.eval()
        pred = self(data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict)
        target = self.ground_truth(data, mode=set)  # .float()
        # pearson = PearsonCorrCoef()
        # loss = pearson(pred, target)
        r2score = R2Score().to(device)
        loss = r2score(pred.flatten(), target.flatten())
        return loss.item()

    @torch.no_grad()
    def test_rmse(self, data, weights=None, set="test"):
        self.eval()
        pred = self(data.x_dict, data.edge_index_dict, self.inspect(data, mode=set), data.edge_attr_dict)
        target = self.ground_truth(data, mode=set)  # .float()
        loss = weighted_mean_absolute_error_loss(pred, target, weight=weights)
        return torch.mean(loss).item()

    @torch.no_grad()
    def test_predict_missings(self, data, dataset, weights=None, penalize_fn=False):
        self.eval()
        missing_idx = self.inspect(data, mode="missing")
        all_preds = self(data.x_dict, data.edge_index_dict, missing_idx, data.edge_attr_dict)
        missing2chembl = dataset.get_missing_in_chembl(missing_idx, edge_type=(self.link_from, self.link_to))
        pred = all_preds[list(missing2chembl.keys())].detach().cpu()
        target = torch.Tensor(list(missing2chembl.values()))
        loss = self.calculate_loss(
            pred.flatten(), target.flatten(), weights=weights, penalize_fn=penalize_fn, reduction="mean"
        )
        return pred, loss, all_preds
