import copy
import itertools as it
import logging
import math
import os
import random
import sys
from statistics import median

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch_geometric.transforms as T
from rdkit import Chem
from torch.distributions.uniform import Uniform
from torch_geometric.data import Dataset, HeteroData
from tqdm.auto import tqdm

from src.biological_properties import (
    generate_esm_encodings,
    get_nb_protein_features,
    get_protein_features,
)
from src.chemical_properties import (
    check_compounds_smiles,
    check_compounds_smiles_from_PACE,
    compounds_similarity,
    get_compound_features,
    get_nb_compound_features,
    get_scaffolds,
)
from src.external import download_uniprot_fasta, retrieve_protein_classes
from src.utils import dataframe_raw_vectorization_with_numpy

# basic logging config
if not os.path.isdir("outputs"):
    os.makedirs("outputs")
logging.basicConfig(filename="outputs/data.log", level=logging.DEBUG)

"""
Randomly assign edges of an heterogeneous graph to train/val/test set
"""


def random_edge_split_heterogeneous(data, edge_type, val_perc, test_perc):
    assert val_perc + test_perc < 1.0
    train_ratio = 1.0 - val_perc - test_perc
    val_ratio = train_ratio + val_perc

    # Keep only edges with an existing label - if there is some "missing" labels
    valid_edges = torch.where(~torch.isnan(data[edge_type].y.flatten()))[0]

    perm = torch.randperm(valid_edges.shape[0])
    # Attach current split as attribute of the data object
    data[edge_type].train_idx = valid_edges[perm[: int(train_ratio * valid_edges.shape[0])]]
    data[edge_type].val_idx = valid_edges[
        perm[int(train_ratio * valid_edges.shape[0]) : int(val_ratio * valid_edges.shape[0])]
    ]
    data[edge_type].test_idx = valid_edges[perm[int(val_ratio * valid_edges.shape[0]) :]]
    return data


"""
Randomly assign edges of an heterogeneous graph to train/val/test set, via a stratified split over the first type
of node (T nodes for T->S edges) attributing the same proportion of train/val/test edges for each node
"""


def stratified_random_edge_split_heterogeneous(data, edge_type, val_perc, test_perc, exclude_singular=True):
    assert val_perc + test_perc < 1.0
    train_ratio = 1.0 - val_perc - test_perc
    val_ratio = train_ratio + val_perc

    nb_nodes = data[edge_type[0]].x.shape[0]

    train_idx = []
    val_idx = []
    test_idx = []
    for node_idx in range(nb_nodes):
        # Check which edge is attached to this node
        node_edges_idx = torch.where(data[edge_type].y_edge_index[0, :] == node_idx)[0]
        # Keep only edges with an existing label - if there is some "missing" labels
        valid_edges = node_edges_idx[torch.where(~torch.isnan(data[edge_type].y[node_edges_idx].flatten()))[0]]

        if exclude_singular:
            # Do not use nodes with only one edge, as it will be unfair
            if len(valid_edges) <= 1:
                continue

        perm = torch.randperm(valid_edges.shape[0])
        train_idx.extend(valid_edges[perm[: int(train_ratio * valid_edges.shape[0])]])
        val_idx.extend(
            valid_edges[perm[int(train_ratio * valid_edges.shape[0]) : int(val_ratio * valid_edges.shape[0])]]
        )
        test_idx.extend(valid_edges[perm[int(val_ratio * valid_edges.shape[0]) :]])
    # Attach current split as attribute of the data object
    data[edge_type].train_idx = torch.LongTensor(train_idx)
    data[edge_type].val_idx = torch.LongTensor(val_idx)
    data[edge_type].test_idx = torch.LongTensor(test_idx)
    return data


"""
Randomly assign edges of an heterogeneous graph to train/val/test set, via a stratified split over the first type of
node (T nodes for T->S edges) attributing the same proportion of train/val/test edges for each node
"""


def stratified_random_train_set_split(data, edge_type, val_perc, test_perc, exclude_singular=False):
    assert test_perc + val_perc < 1.0
    val_ratio = val_perc / (1.0 - test_perc)
    train_ratio = 1.0 - val_ratio

    train_idx = []
    val_idx = []
    valid_edges = torch.LongTensor(
        [idx for idx, val in enumerate(data[edge_type].y) if idx not in data[edge_type].test_idx]
    )

    perm = torch.randperm(valid_edges.shape[0])
    train_idx.extend(valid_edges[perm[: int(train_ratio * valid_edges.shape[0])].tolist()].tolist())
    val_idx.extend(valid_edges[perm[int(train_ratio * valid_edges.shape[0]) :].tolist()].tolist())

    # Attach current split as attribute of the data object
    data[edge_type].train_idx = torch.LongTensor(train_idx)
    data[edge_type].val_idx = torch.LongTensor(val_idx)
    return data


"""
Randomly assign edges of an heterogeneous graph to train/val/test set, via a split by chemical scaffolds
"""


def scaffold_edge_split_heterogeneous(dataset, data, edge_type, val_perc, test_perc, exclude_singular=True):
    assert val_perc + test_perc < 1.0
    train_ratio = 1.0 - val_perc - test_perc
    val_ratio = train_ratio + val_perc

    # Check if compound is the first or the second type of nodes
    if edge_type[0] == "compound":
        idx_compounds = 0
    elif edge_type[1] == "compound":
        idx_compounds = 1
    else:
        raise ValueError("Cannot use scaffold split on edge type '{}'".format(edge_type))

    nb_compounds = len(dataset.compounds)
    # Compute scaffold from compound SMILES
    comp2scaffold, scaffold_idx = get_scaffolds(dataset.comp2smiles)

    perm_scaffold_idx = np.array(list(scaffold_idx.values()))[torch.randperm(len(scaffold_idx)).numpy()]
    nb_scaffolds = len(perm_scaffold_idx)

    train_idx = []
    val_idx = []
    test_idx = []
    for comp_idx in range(nb_compounds):
        # For each node, check which edge is attached to this node
        node_edges_idx = torch.where(data[edge_type].y_edge_index[idx_compounds, :] == comp_idx)[0]
        # Keep only edges with an existing label - if there is some "missing" labels
        valid_edges = node_edges_idx[torch.where(~torch.isnan(data[edge_type].y[node_edges_idx].flatten()))[0]]

        if comp2scaffold[dataset.compounds[comp_idx]] in perm_scaffold_idx[: int(train_ratio * nb_scaffolds)]:
            train_idx.extend(valid_edges)
        elif (
            comp2scaffold[dataset.compounds[comp_idx]]
            in perm_scaffold_idx[int(train_ratio * nb_scaffolds) : int(val_ratio * nb_scaffolds)]
        ):
            val_idx.extend(valid_edges)
        elif comp2scaffold[dataset.compounds[comp_idx]] in perm_scaffold_idx[int(val_ratio * nb_scaffolds) :]:
            test_idx.extend(valid_edges)
        else:
            raise ValueError("Something went wrong during the scaffold split...")
    # Attach current split as attribute of the data object
    data[edge_type].train_idx = torch.LongTensor(train_idx)
    data[edge_type].val_idx = torch.LongTensor(val_idx)
    data[edge_type].test_idx = torch.LongTensor(test_idx)
    return data


"""
Find and return singular edges, i.e. edges from nodes which only have one single attached edge.
"""


def get_singular_edge_indices(data, edge_type):
    nb_nodes = data[edge_type[0]].x.shape[0]
    singular_idx = []
    for node_idx in range(nb_nodes):
        # For each node, check which edge is attached to this node
        node_edges_idx = torch.where(data[edge_type].y_edge_index[0, :] == node_idx)[0]
        # Keep only edges with an existing label - if there is some "missing" labels
        valid_edges = node_edges_idx[torch.where(~torch.isnan(data[edge_type].y[node_edges_idx].flatten()))[0]]
        # Record nodes with only one edge
        if len(valid_edges) == 1:
            singular_idx.append(valid_edges.item())
    return singular_idx


def edge_complete_set_heterogeneous(data, edge_type):
    # Return train set = val set = test set to overfit model
    perm = torch.randperm(data[edge_type].num_edges)
    # Attach current split as attribute of the data object
    data[edge_type].train_idx = perm
    data[edge_type].val_idx = perm
    data[edge_type].test_idx = perm
    return data


"""
Create edge batches from the train set
"""


class GraphBatchIterator:
    def __init__(self, data, edge_type, batch_size, use_dtis):
        complete_train_idx = data[edge_type].train_idx.detach().clone()

        rev_edge_type = (
            edge_type[1],
            edge_type[0],
        )  # We need to use edges in both directions in order to correctly pull information

        self.batches = []
        for b_idx in range(0, len(complete_train_idx), batch_size):
            batch = data.cpu().detach().clone()  # Create copy of class data
            batch[edge_type].train_idx = complete_train_idx[
                b_idx : b_idx + batch_size
            ]  # Keep only edges belonging to the current batch as train indices
            if use_dtis:
                # Add other train idx (the ones left out from the batch) as real edges on the graph
                known_idx = torch.cat(
                    (complete_train_idx[0:b_idx], complete_train_idx[b_idx + batch_size : len(complete_train_idx)]),
                    dim=0,
                )
                batch[edge_type].edge_index = batch[edge_type].y_edge_index[:, known_idx].detach().clone()
                batch[edge_type].edge_attr = batch[edge_type].y[known_idx, :].detach().clone()

                batch[rev_edge_type].edge_index = (
                    batch[edge_type].y_edge_index[:, known_idx].flip(dims=[0]).detach().clone()
                )
                batch[rev_edge_type].edge_attr = batch[edge_type].y[known_idx, :].detach().clone()

            self.batches.append(batch)
        self.nb_batches = len(self.batches)
        self.batch_size = batch_size
        self.use_dtis = use_dtis

    def __iter__(self):
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= self.nb_batches:
            raise StopIteration
        else:
            self.batch_idx += 1  # Increment before returning value
            return self.batches[self.batch_idx - 1]


def batch_train_edges(data, edge_type, batch_size):
    complete_train_idx = data[edge_type].train_idx.detach().clone()

    batched_data = []
    for b_idx in range(0, len(complete_train_idx), batch_size):
        batch = data.detach().clone()  # Create copy of class data
        batch[edge_type].train_idx = complete_train_idx[
            b_idx : b_idx + batch_size
        ]  # Keep only edges belonging to the current batch as train indices
        batched_data.append(batch)
    return batched_data


"""
Get edges without labels
"""


def get_missing_links(data, edge_type):
    missing_edge_indices = []
    edge_from, edge_to = edge_type[0], edge_type[1]

    for from_idx in range(data[edge_from].num_nodes):
        for to_idx in range(data[edge_to].num_nodes):
            # Check if a labelized edge exists
            existing = torch.where(
                (data[edge_type].y_edge_index[0, :] == from_idx) & (data[edge_type].y_edge_index[1, :] == to_idx)
            )[0]
            if len(existing) == 0:  # No labelized edge
                missing_edge_indices.append([from_idx, to_idx])
            elif len(existing) > 1:  # Duplicate - shouldn't happen if the graph is built correctly
                logging.error("Graph Dataset: Duplicated edges found during missing links extraction!")
    # Transpose the array to have the correct shape (2, Nb_edges)
    missing_edge_indices = np.array(missing_edge_indices).T
    return torch.LongTensor(missing_edge_indices)


class PDADataset(Dataset):
    def __init__(self, settings):
        self.data_folder = settings.RUNINFO.data_dir
        self.preprocessed_data_path = os.path.join(self.data_folder, "preprocessed.json")
        self.settings = settings  # Contains all the customized setting
        super().__init__("./", transform=None, pre_filter=None, pre_transform=None)

    # Create common and correct name to save the data, from the defined setting
    @property
    def processed_file_names(self):
        base = "pda-data"

        if self.settings.INPUT_FILES.average_screenings:
            # In case several screenings are available for the same PLI
            base = base + "_Averaged"

        if self.settings.COMPOUND_FEATURES.fingerprints:
            base = base + "_Fingerprints{}-{}".format(
                self.settings.COMPOUND_FEATURES.fingerprints_type, self.settings.COMPOUND_FEATURES.fingerprints_size
            )
        if self.settings.COMPOUND_FEATURES.lipinski_features:
            base = base + "_LipinskiFea"
        if self.settings.COMPOUND_FEATURES.extended_chemical_features:
            base = base + "_ExtChemicalFea"

        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            comp_sim_value = (
                str(self.get_compound_similarity_threshold())
                if self.get_compound_similarity_threshold() != 0.0
                else ""
            )
            if self.settings.COMPOUND_FEATURES.compound_similarities:
                base = base + "_CompSimilarities{}".format(comp_sim_value)
            else:
                base = base + "_NoCompSimilarities"
        else:
            base = base + "_NoCompLinks"

        if self.settings.PROTEIN_FEATURES.khmer:
            base = base + "_{}-khmer".format(self.settings.PROTEIN_FEATURES.khmer_size)
        if self.settings.PROTEIN_FEATURES.gene_expression:
            base = base + "_GeneExpr"
            if (
                self.settings.PROTEIN_FEATURES.gene_expression_tissue is None
                or self.settings.PROTEIN_FEATURES.gene_expression_tissue.lower() == "all"
            ):
                base = base + "All"
            else:
                base = base + self.settings.PROTEIN_FEATURES.gene_expression_tissue.title()
        if self.settings.PROTEIN_FEATURES.subcellular_location:
            base = base + "_SubcellularLoc"
        if self.settings.PROTEIN_FEATURES.esm_encoding:
            base = base + "_{}".format(self.settings.PROTEIN_FEATURES.esm_encoding)

        if self.settings.PROTEIN_FEATURES.protein_protein_interactions:
            base = base + "_{}".format(self.settings.INPUT_FILES.ppi_database.replace("_ppi.csv", ""))
        else:
            base = base + "_noPPI"

        # Add PPI database used in the save name
        return base + ".pt"

    def load_data(self):
        self.data = torch.load(os.path.join(self.data_folder, self.processed_file_names))
        self.targets = self.data["target"]["names"]
        self.compounds = self.data["compound"]["names"]
        return self.data

    def read_data(self):
        # Read dataset files
        pace_annot_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.pace_annot), delimiter="\t"
        )
        pace_res_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.pace_library),
            delimiter="\t",
            encoding="windows-1252",
        )
        bds_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.bds_screenings), delimiter="\t", low_memory=False
        )
        chembl_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.chembl_data),
            delimiter="\t",
            encoding="windows-1252",
        )
        ppi_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.ppi_database),
            header=None,
            names=["Prot1", "Prot2"],
        )
        gene_expr_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.gene_expr_database), delimiter="\t"
        )

        # Load mapping file
        mapping_file = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.mapping_file), delimiter="\t"
        )

        # Exclude BDS with unclear target
        logging.warning(
            "PDA Dataset: Excluding {} BDS results with unclear target GeneID...".format(
                bds_df.TARGET_GENEID.isna().sum()
            )
        )
        bds_df = bds_df[~bds_df.TARGET_GENEID.isna()]

        # Download .fasta protein sequences
        download_uniprot_fasta(self.targets, self.data_folder)

        self.targets, excluded_targets = self.check_ensembl_id(
            self.targets, mapping_file
        )  # Check if every target have a correct EnsemblGeneID, exclude otherwise
        self.targets, excluded_targets = self.check_proteins_validity(
            self.targets, mapping_file, gene_expr_df
        )  # Check if every target is successfully compiled, exclude otherwise

        # Retrieve diverse protein classes from UniProt
        self.target_classes = retrieve_protein_classes(self.targets)

        # Register SMILES for each of the compounds
        comp2smiles = {comp: [] for comp in bds_df.CORE.unique()}
        for comp, smiles in dataframe_raw_vectorization_with_numpy(bds_df, keys=["CORE", "SMILES"]):
            if not (isinstance(smiles, float) and math.isnan(smiles)):
                comp2smiles[comp].append(smiles)

        # List unique compounds
        self.compounds = bds_df.CORE.unique().tolist()
        # Exclude compounds with an unvalid SMILES
        if self.settings.INPUT_FILES.restrict_to_pace_compounds:
            self.compounds, self.comp2smiles = check_compounds_smiles_from_PACE(
                pace_res_df, self.compounds
            )  # Compounds not in the PACE library are excluded here
        else:
            self.compounds, self.comp2smiles = check_compounds_smiles(comp2smiles, self.compounds)

        # Filter data to keep only relevant rows
        gene_ids = mapping_file[mapping_file.Accession.isin(self.targets)].GeneID.unique()
        bds_df = bds_df[(bds_df.TARGET_GENEID.isin(gene_ids)) & (bds_df.CORE.isin(self.compounds))]

        return pace_annot_df, pace_res_df, bds_df, chembl_df, ppi_df, mapping_file, gene_expr_df

    def check_ensembl_id(self, targets, mapping_file):
        excluded_targets = []
        for target in targets:
            # Check if each target has at least one matching EnsemblGeneID
            target_ensembl = mapping_file[mapping_file.Accession == target].EnsemblGeneID.unique().tolist()
            if len(target_ensembl) == 0:
                excluded_targets.append(target)
        logging.warning(
            "Dataset: The following targets were excluded from the pipeline as lacking an EnsemblID:\n{}\n".format(
                excluded_targets
            )
        )
        return [t for t in targets if t not in excluded_targets], excluded_targets

    def check_proteins_validity(self, targets, mapping_file, gene_expr_df):
        # Exclude targets when unable to compute protein features. Such exception are mainly due to obsolete
        # Uniprot ID or miscalleneous fasta file (ex: B1ALC3)
        # Computationaly redundant, but we need to exclude those targets before building the graph
        excluded_targets = []
        target_esm_encodings = generate_esm_encodings(targets)
        for target in tqdm(targets, desc="Checking correctness of targets..."):
            target_ensembl = mapping_file[mapping_file.Accession == target].EnsemblGeneID.unique().tolist()
            try:
                _ = get_protein_features(
                    target,
                    target_ensembl,
                    gene_expr_df,
                    target_esm_encodings,
                    self.data_folder,
                    khmer=self.settings.PROTEIN_FEATURES.khmer,
                    k_khmer=self.settings.PROTEIN_FEATURES.khmer_size,
                    sublocation=self.settings.PROTEIN_FEATURES.subcellular_location,
                    gene_expr=self.settings.PROTEIN_FEATURES.gene_expression,
                    esm_encoding=self.settings.PROTEIN_FEATURES.esm_encoding,
                )
            except Exception:
                logging.warning("Cannot compute for target {}".format(target))
                excluded_targets.append(target)
        logging.warning(
            "Dataset: The following targets were excluded from the pipeline as the program was unable to"
            + "compute protein features:\n{}\n".format(excluded_targets)
        )
        return [t for t in targets if t not in excluded_targets], excluded_targets

    def build_graph(self):
        # Read data
        pace_annot_df, pace_res_df, bds_df, chembl_df, ppi_df, mapping_file, gene_expr_df = self.read_data()

        print(
            "Initiating a graph containing {} compounds and {} targets...".format(
                len(self.compounds), len(self.targets)
            )
        )

        # Computation of protein features
        # We advice k_khmer to be 3: catalytic site and its neighbours
        target_features = np.empty(
            (
                len(self.targets),
                get_nb_protein_features(
                    khmer=self.settings.PROTEIN_FEATURES.khmer,
                    k_khmer=self.settings.PROTEIN_FEATURES.khmer_size,
                    sublocation=self.settings.PROTEIN_FEATURES.subcellular_location,
                    gene_expr=self.settings.PROTEIN_FEATURES.gene_expression,
                    esm_encoding=self.settings.PROTEIN_FEATURES.esm_encoding,
                ),
            ),
            dtype=np.float32,
        )
        target_esm_encodings = generate_esm_encodings(self.targets)
        for idx, target in tqdm(enumerate(self.targets), file=sys.stdout, desc="Calculate protein features..."):
            target_ensembl = mapping_file[mapping_file.Accession == target].EnsemblGeneID.unique().tolist()
            target_features[idx, :] = get_protein_features(
                target,
                target_ensembl,
                gene_expr_df,
                target_esm_encodings,
                self.data_folder,
                khmer=self.settings.PROTEIN_FEATURES.khmer,
                k_khmer=self.settings.PROTEIN_FEATURES.khmer_size,
                sublocation=self.settings.PROTEIN_FEATURES.subcellular_location,
                gene_expr=self.settings.PROTEIN_FEATURES.gene_expression,
                esm_encoding=self.settings.PROTEIN_FEATURES.esm_encoding,
                gene_expr_tissue=self.settings.PROTEIN_FEATURES.gene_expression_tissue,
            )

        # Computation of chemical features
        fingerprint = (
            self.settings.COMPOUND_FEATURES.fingerprints_type if self.settings.COMPOUND_FEATURES.fingerprints else None
        )
        compound_features = np.empty(
            (
                len(self.compounds),
                get_nb_compound_features(
                    fingerprint=fingerprint,
                    fingerprint_size=self.settings.COMPOUND_FEATURES.fingerprints_size,
                    lipinski_features=self.settings.COMPOUND_FEATURES.lipinski_features,
                    other_features=self.settings.COMPOUND_FEATURES.extended_chemical_features,
                ),
            ),
            dtype=np.float32,
        )
        for idx, comp in tqdm(enumerate(self.compounds), file=sys.stdout, desc="Calculate compound features..."):
            compound_smiles = self.comp2smiles[comp]
            compound_features[idx, :] = get_compound_features(
                compound_smiles,
                to_list=True,
                fingerprint=fingerprint,
                fingerprint_size=self.settings.COMPOUND_FEATURES.fingerprints_size,
                lipinski_features=self.settings.COMPOUND_FEATURES.lipinski_features,
                other_features=self.settings.COMPOUND_FEATURES.extended_chemical_features,
            )

        # Edges for compound-targets interactions
        edges_inter_from_to = []
        edges_inter_y = []

        # Create a dict of experimental pAct
        edge_inter_dict = {prot: {compound: [] for compound in self.compounds} for prot in self.targets}
        # Get pAct info
        for target_gene_id, compound_name, pAct in tqdm(
            dataframe_raw_vectorization_with_numpy(bds_df, keys=["TARGET_GENEID", "CORE", "pAct"]),
            file=sys.stdout,
            desc="Edges pAct",
        ):
            target_uniprot_ids = mapping_file[mapping_file.GeneID == int(target_gene_id)].Accession.to_list()

            # We are using a PPI network referencing gene IDs
            # -> Translate them to UniProt IDs (there can be several proteins!)
            for uniprot_id in target_uniprot_ids:
                if compound_name in self.compounds and uniprot_id in self.targets:
                    edge_inter_dict[uniprot_id][compound_name].append(pAct)
                else:
                    logging.warning(
                        "PDADataset: pAct recorded on unknown drug or target (compound {} - target {})".format(
                            compound_name, uniprot_id
                        )
                    )

        for uniprot_id in self.targets:
            for compound_name in self.compounds:
                if edge_inter_dict[uniprot_id][compound_name]:  # list is not empty
                    edges_inter_from_to.append([compound_name, uniprot_id])
                    if self.settings.INPUT_FILES.average_screenings:
                        # Average (by median) and fill in edges
                        edges_inter_y.append([float(median(edge_inter_dict[uniprot_id][compound_name]))])
                    else:
                        # Pick randomly one of the pAct and fill in edges
                        edges_inter_y.append([random.choice(edge_inter_dict[uniprot_id][compound_name])])

        # Edges for protein-protein interactions
        if self.settings.PROTEIN_FEATURES.protein_protein_interactions:
            edges_ppi_from_to = []
            edges_ppi_features = []

            for prot1_geneid, prot2_geneid in dataframe_raw_vectorization_with_numpy(ppi_df, keys=["Prot1", "Prot2"]):
                # We are using a PPI network referencing gene IDs
                # -> Translate them to UniProt IDs (there can be several proteins!)
                prot1_ids = mapping_file[mapping_file.GeneID == prot1_geneid].Accession.to_list()
                prot2_ids = mapping_file[mapping_file.GeneID == prot2_geneid].Accession.to_list()
                for prot1 in prot1_ids:
                    for prot2 in prot2_ids:
                        if prot1 in self.targets and prot2 in self.targets:
                            if [prot1, prot2] not in edges_ppi_from_to:
                                edges_ppi_from_to.append([prot1, prot2])
                                edges_ppi_features.append([1])  # TODO: Improve by providing Confidence Score here
                            else:
                                logging.warning("PDA Dataset: Trying to duplicate edge ({}, {})".format(prot1, prot2))

        # Edges for compound-compound interaction
        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:  # If none, compounds are not linked
            similarity_threshold = self.get_compound_similarity_threshold()

            edges_drugs_from_to = []
            edges_drugs_features = []

            for comp1, comp2 in it.combinations(self.compounds, r=2):
                if (
                    self.settings.COMPOUND_FEATURES.compound_similarities
                ):  # If True, compounds are linked by compound similarities
                    # Use compound similarities on the edges
                    similarity = compounds_similarity(self.comp2smiles[comp1], self.comp2smiles[comp2])
                    if similarity >= similarity_threshold:
                        edges_drugs_from_to.append([comp1, comp2])
                        edges_drugs_features.append(similarity)
                else:  # If False, all compounds are linked by dummy links
                    edges_drugs_from_to.append([comp1, comp2])
                    edges_drugs_features.append([1.0])

        # Format data into numpy arrays
        target_features = np.array(target_features)
        compound_features = np.array(compound_features)
        edges_inter_from_to = np.array(
            [
                [self.compounds.index(edge_from) for edge_from, _ in edges_inter_from_to],
                [self.targets.index(edge_to) for _, edge_to in edges_inter_from_to],
            ]
        )
        if self.settings.PROTEIN_FEATURES.protein_protein_interactions:
            edges_ppi_from_to = np.array(
                [
                    [self.targets.index(edge_from) for edge_from, _ in edges_ppi_from_to],
                    [self.targets.index(edge_to) for _, edge_to in edges_ppi_from_to],
                ]
            )
            edges_ppi_features = np.array(edges_ppi_features)

        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            # Draw edges unless 'None' compound similarities
            edges_drugs_from_to = np.array(
                [
                    [self.compounds.index(edge_from) for edge_from, _ in edges_drugs_from_to],
                    [self.compounds.index(edge_to) for _, edge_to in edges_drugs_from_to],
                ]
            )
            edges_drugs_features = np.array(edges_drugs_features)

        # Create the graph
        data = HeteroData()

        # Fill in nodes and edges following the PyTorchGeometric structure
        data["target"].x = torch.Tensor(target_features)
        data["target"].names = self.targets
        data["compound"].x = torch.Tensor(compound_features)
        data["compound"].names = self.compounds
        data["compound"].smiles = [self.comp2smiles[k] for k in self.compounds]

        if self.settings.PROTEIN_FEATURES.protein_protein_interactions:
            data["target", "interaction", "target"].edge_index = torch.LongTensor(edges_ppi_from_to)
            data["target", "interaction", "target"].edge_attr = torch.Tensor(edges_ppi_features)
        else:
            data["target", "interaction", "target"].edge_index = torch.ones((2, 0)).long()
            data["target", "interaction", "target"].edge_attr = torch.ones((0, 1)).float()

        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            data["compound", "similarity", "compound"].edge_index = torch.LongTensor(edges_drugs_from_to)
            data["compound", "similarity", "compound"].edge_attr = torch.Tensor(edges_drugs_features)
        else:
            data["compound", "similarity", "compound"].edge_index = torch.ones((2, 0)).long()
            data["compound", "similarity", "compound"].edge_attr = torch.ones((0, 1)).float()

        # Set label to predict as pAct for protein-drug interaction - those interactions are not part of graph topology
        data["compound", "interaction", "target"].edge_index = torch.ones(
            (2, 0)
        ).long()  # Create empty edge index - for compatibility
        data["compound", "interaction", "target"].edge_attr = torch.ones(
            (0, 1)
        ).float()  # Create empty edge attr - for compatibility
        data["compound", "interaction", "target"].y_edge_index = torch.LongTensor(edges_inter_from_to)
        data["compound", "interaction", "target"].y = torch.Tensor(edges_inter_y)

        # Make graph undirected
        data = self.transform_graph(data)

        print("Graph initialized!\n{}\n\n".format(data))
        return data

    def add_target_nodes(self, node_names):
        ppi_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.ppi_database),
            header=None,
            names=["Prot1", "Prot2"],
        )

        valid_node_names = []
        for node_name in node_names:
            if node_name in self.data["target"].names:
                logging.warning("PDA Dataset: Trying to duplicate node {}".format(node_name))
            else:
                valid_node_names.append(node_name)

        if len(valid_node_names) == 0:
            return []

        mapping_file = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.mapping_file), delimiter="\t"
        )
        gene_expr_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.gene_expr_database), delimiter="\t"
        )

        download_uniprot_fasta(valid_node_names, self.data_folder)

        temp_node_names, excluded_targets = self.check_ensembl_id(
            valid_node_names, mapping_file
        )  # Check if every target have a correct EnsemblGeneID, exclude otherwise
        if len(temp_node_names) == 0:
            logging.error("PDA Dataset: Target {} not found in mapping file".format(excluded_targets))
            return None
        valid_node_names, excluded_targets = self.check_proteins_validity(
            temp_node_names, mapping_file, gene_expr_df
        )  # Check if every target is successfully compiled, exclude otherwise
        if len(valid_node_names) == 0:
            logging.error("PDA Dataset: Could not compute Target {}'s features".format(excluded_targets))
            return None

        try:
            # Retrieve diverse protein classes from UniProt
            self.target_classes.append(retrieve_protein_classes([node_name for node_name in valid_node_names]))
        except Exception:
            pass

        target_features = np.empty(
            (
                len(valid_node_names),
                get_nb_protein_features(
                    khmer=self.settings.PROTEIN_FEATURES.khmer,
                    k_khmer=self.settings.PROTEIN_FEATURES.khmer_size,
                    sublocation=self.settings.PROTEIN_FEATURES.subcellular_location,
                    gene_expr=self.settings.PROTEIN_FEATURES.gene_expression,
                    esm_encoding=self.settings.PROTEIN_FEATURES.esm_encoding,
                ),
            ),
            dtype=np.float32,
        )
        target_esm_encodings = generate_esm_encodings(valid_node_names)
        for idx, target in tqdm(enumerate(valid_node_names), file=sys.stdout, desc="Calculate protein features..."):
            target_ensembl = mapping_file[mapping_file.Accession == target].EnsemblGeneID.unique().tolist()
            target_features[idx, :] = get_protein_features(
                target,
                target_ensembl,
                gene_expr_df,
                target_esm_encodings,
                self.data_folder,
                khmer=self.settings.PROTEIN_FEATURES.khmer,
                k_khmer=self.settings.PROTEIN_FEATURES.khmer_size,
                sublocation=self.settings.PROTEIN_FEATURES.subcellular_location,
                gene_expr=self.settings.PROTEIN_FEATURES.gene_expression,
                esm_encoding=self.settings.PROTEIN_FEATURES.esm_encoding,
                gene_expr_tissue=self.settings.PROTEIN_FEATURES.gene_expression_tissue,
            )
        self.data["target"].x = torch.cat((self.data["target"].x, torch.from_numpy(target_features)), dim=0)
        self.data["target"].names = self.data["target"].names + valid_node_names
        self.targets = self.targets + valid_node_names
        assert len(self.targets) == len(np.unique(self.targets)), "PDA Dataset: Duplicated targets in dataset"

        # Extend protein-protein interaction network
        if self.settings.PROTEIN_FEATURES.protein_protein_interactions:
            edges_ppi_from_to = []
            edges_ppi_features = []

            for prot1_geneid, prot2_geneid in tqdm(
                dataframe_raw_vectorization_with_numpy(ppi_df, keys=["Prot1", "Prot2"]),
                file=sys.stdout,
                desc="Updating protein-protein interactions...",
            ):
                # We are using a PPI network referencing gene IDs
                # -> Translate them to UniProt IDs (there can be several proteins!)
                prot1_ids = mapping_file[mapping_file.GeneID == prot1_geneid].Accession.to_list()
                prot2_ids = mapping_file[mapping_file.GeneID == prot2_geneid].Accession.to_list()
                for prot1 in prot1_ids:
                    for prot2 in prot2_ids:
                        if prot1 in valid_node_names or prot2 in valid_node_names:  # Concern new proteins
                            if (
                                prot1 in self.targets and prot2 in self.targets
                            ):  # Link between two proteins in our graph
                                if [prot1, prot2] not in edges_ppi_from_to:
                                    edges_ppi_from_to.append([prot1, prot2])
                                    edges_ppi_features.append([1])  # TODO: Improve by providing Confidence Score here
                                else:
                                    logging.warning(
                                        "PDA Dataset: Trying to duplicate edge ({}, {})".format(prot1, prot2)
                                    )
            self.data["target", "interaction", "target"].edge_index = torch.LongTensor(
                [
                    self.data["target", "interaction", "target"].edge_index[0].tolist()
                    + [self.targets.index(edge_from) for edge_from, _ in edges_ppi_from_to],
                    self.data["target", "interaction", "target"].edge_index[1].tolist()
                    + [self.targets.index(edge_to) for _, edge_to in edges_ppi_from_to],
                ]
            )
            self.data["target", "interaction", "target"].edge_attr = torch.cat(
                (self.data["target", "interaction", "target"].edge_attr, torch.Tensor(edges_ppi_features)), dim=0
            )

        return valid_node_names

    def add_compound_nodes(self, node_names, smiles_dict=None):
        valid_node_names = []
        for node_name in node_names:
            if node_name in self.data["compound"].names or node_name in self.data["compound"].smiles:
                logging.warning("PDA Dataset: Trying to duplicate node {}".format(node_name))
            else:
                try:
                    if smiles_dict is not None:
                        if node_name in smiles_dict:
                            Chem.MolFromSmiles(smiles_dict[node_name])
                            valid_node_names.append(smiles_dict[node_name])
                        else:
                            logging.warning("PDA Dataset: Trying to add node {} with no SMILES".format(node_name))
                            continue
                    else:
                        Chem.MolFromSmiles(node_name)
                        valid_node_names.append(node_name)
                except Exception:
                    logging.warning("PDA Dataset: Trying to add invalid SMILES {}".format(node_name))
                    continue

        # Computation of chemical features
        fingerprint = (
            self.settings.COMPOUND_FEATURES.fingerprints_type if self.settings.COMPOUND_FEATURES.fingerprints else None
        )
        compound_features = np.empty(
            (
                len(valid_node_names),
                get_nb_compound_features(
                    fingerprint=fingerprint,
                    fingerprint_size=self.settings.COMPOUND_FEATURES.fingerprints_size,
                    lipinski_features=self.settings.COMPOUND_FEATURES.lipinski_features,
                    other_features=self.settings.COMPOUND_FEATURES.extended_chemical_features,
                ),
            ),
            dtype=np.float32,
        )
        for idx, comp in tqdm(enumerate(valid_node_names), file=sys.stdout, desc="Calculate compound features..."):
            try:
                self.comp2smiles[comp] = comp
            except Exception:
                pass

            compound_features[idx, :] = get_compound_features(
                comp,
                to_list=True,
                fingerprint=fingerprint,
                fingerprint_size=self.settings.COMPOUND_FEATURES.fingerprints_size,
                lipinski_features=self.settings.COMPOUND_FEATURES.lipinski_features,
                other_features=self.settings.COMPOUND_FEATURES.extended_chemical_features,
            )
        self.data["compound"].x = torch.cat((self.data["compound"].x, torch.from_numpy(compound_features)), dim=0)
        self.compounds = self.compounds + valid_node_names
        assert len(self.compounds) == len(np.unique(self.compounds)), "PDA Dataset: Duplicated compounds in dataset"
        self.data["compound"].names = self.data["compound"].names + valid_node_names
        self.data["compound"].smiles = self.data["compound"].smiles + [comp for comp in valid_node_names]

        # For new nodes, add edges for compound-compound interaction
        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:  # If none, compounds are not linked
            similarity_threshold = self.get_compound_similarity_threshold()

            edges_drugs_from_to = []
            edges_drugs_features = []

            for comp1 in tqdm(valid_node_names, file=sys.stdout, desc="Calculate compound similarities..."):
                # Edges new to all compounds (including new)
                for comp2 in self.data["compound"].smiles:
                    if comp1 != comp2:
                        if (
                            self.settings.COMPOUND_FEATURES.compound_similarities
                        ):  # If True, compounds are linked by compound similarities
                            # Use compound similarities on the edges
                            try:
                                similarity = compounds_similarity(self.comp2smiles[comp1], self.comp2smiles[comp2])
                            except Exception:
                                similarity = compounds_similarity(comp1, comp2)
                            if similarity >= similarity_threshold:
                                edges_drugs_from_to.append([comp1, comp2])
                                edges_drugs_features.append(similarity)
                        else:  # If False, all compounds are linked by dummy links
                            edges_drugs_from_to.append([comp1, comp2])
                            edges_drugs_features.append([1.0])

                        # Draw edges unless 'None' compound similarities
            self.data["compound", "similarity", "compound"].edge_index = torch.LongTensor(
                [
                    self.data["compound", "similarity", "compound"].edge_index[0].tolist()
                    + [self.data["compound"].smiles.index(edge_from) for edge_from, _ in edges_drugs_from_to],
                    self.data["compound", "similarity", "compound"].edge_index[1].tolist()
                    + [self.data["compound"].smiles.index(edge_to) for _, edge_to in edges_drugs_from_to],
                ]
            )
            self.data["compound", "similarity", "compound"].edge_attr = torch.cat(
                (self.data["compound", "similarity", "compound"].edge_attr, torch.Tensor(edges_drugs_features)), dim=0
            )

        return valid_node_names

    def add_labeled_edges(self, compound_node_names, target_node_names, edge_labels):
        assert (
            len(compound_node_names) == len(target_node_names) == len(edge_labels)
        ), "PDA Dataset: Invalid number of edges"

        existing_edges = [(edge[0].item(), edge[1].item()) for edge in self.data["compound", "target"].y_edge_index.T]
        valid_edges = []
        valid_labels = []
        for compound, target, label in zip(compound_node_names, target_node_names, edge_labels):
            if compound in self.data["compound"].names and target in self.data["target"].names:
                compound_idx = self.data["compound"].names.index(compound)
                target_idx = self.data["target"].names.index(target)
                if (compound_idx, target_idx) not in existing_edges:
                    valid_edges.append([compound_idx, target_idx])
                    valid_labels.append([label])
                else:
                    logging.warning("PDA Dataset: Edge {}-{} not added".format(compound_idx, target_idx))
        if len(valid_edges) != 0:
            self.data["compound", "target"].y_edge_index = torch.cat(
                (self.data["compound", "target"].y_edge_index, torch.LongTensor(valid_edges).T), dim=1
            )
            self.data["compound", "target"].y = torch.cat(
                (self.data["compound", "target"].y, torch.Tensor(valid_labels)), dim=0
            )
            self.data["target", "compound"].y_edge_index = torch.cat(
                (
                    self.data["target", "compound"].y_edge_index,
                    torch.LongTensor([edge[::-1] for edge in valid_edges]).T,
                ),
                dim=1,
            )
            self.data["target", "compound"].y = torch.cat(
                (self.data["target", "compound"].y, torch.Tensor(valid_labels)), dim=0
            )

        return valid_edges

    def get_compound_similarity_threshold(self):
        if (
            self.settings.COMPOUND_FEATURES.compound_similarities_threshold >= 0
            and self.settings.COMPOUND_FEATURES.compound_similarities_threshold < 1
        ):
            return self.settings.COMPOUND_FEATURES.compound_similarities_threshold
        else:
            logging.warning(
                "PDA Dataset: Compound similarity threshold is expected to be between 0.0 and 1.0 (default 0.0)"
            )
            return 0.0

    def transform_graph(self, data):
        # Make graph undirected - PyTorchGeometric performs the transformation by adding reverse edges
        # (on every heterogeneous relations)
        data = T.ToUndirected()(data)
        # Manually copy labels to reverted edges
        data["target", "compound"].y_edge_index = torch.LongTensor(
            [
                data["compound", "target"].y_edge_index[1, :].tolist(),
                data["compound", "target"].y_edge_index[0, :].tolist(),
            ]
        )
        data["target", "compound"].y = torch.Tensor(data["compound", "target"].y.tolist())
        return data

    def randomize_edges(self, seed=123, random_pAct=True):
        # Function build to randomize edges and their label, and return a 'random graph' in order to compare learning
        # processes and biological understanding of the model.
        # Here we are drawing bioactivities values randomly, instead of shuffling the existing ones.
        torch.manual_seed(seed)

        data = copy.deepcopy(self.data)

        nb_edges = data["compound", "target"].y.shape[0]  # Keep the same number of edges

        # Either pick random pAct (following a uniform distribution) or shuffling existing pAct
        if random_pAct:
            y_perm = Uniform(low=3.0, high=11.0).sample((nb_edges, 1))
        else:
            y_perm = data["compound", "target"].y[torch.randperm(nb_edges)]

        # Randomly choose nodes in relation
        data["compound", "target"].y_edge_index = torch.LongTensor(
            [
                torch.randint(low=0, high=len(self.compounds), size=(nb_edges,)).tolist(),
                torch.randint(low=0, high=len(self.targets), size=(nb_edges,)).tolist(),
            ]
        )
        data["compound", "target"].y = y_perm

        # Manually copy labels to reverted edges
        data["target", "compound"].y_edge_index = torch.LongTensor(
            [
                data["compound", "target"].y_edge_index[1, :].tolist(),
                data["compound", "target"].y_edge_index[0, :].tolist(),
            ]
        )
        data["target", "compound"].y = torch.Tensor(data["compound", "target"].y.tolist())

        return data

    def build_process(self):
        # Build and save graph
        self.data = self.build_graph()
        torch.save(self.data, os.path.join(self.data_folder, self.processed_file_names))
        return self.data

    def len(self):
        return len(self.processed_file_names)

    def get(self):
        # Build graph, or load it if it has already been saved
        if os.path.isfile(os.path.join(self.data_folder, self.processed_file_names)):
            print(
                "PDA Dataset: Loading data from {} ...".format(
                    os.path.join(self.data_folder, self.processed_file_names)
                )
            )
            self.data = self.load_data()
        else:
            print("PDA Dataset: Building data...")
            self.data = self.build_process()
        return self.data

    def get_metadata(self, use_dtis):
        # Custom metadata getter, because convolutional layers shouldn't use all target-compound for message passing
        if use_dtis:  # Here we'll use the target-compounds in the train set
            return (
                ["target", "compound"],
                [
                    ("target", "interaction", "target"),
                    ("compound", "similarity", "compound"),
                    ("compound", "interaction", "target"),
                    ("target", "rev_interaction", "compound"),
                ],
            )
        else:
            return (
                ["target", "compound"],
                [("target", "interaction", "target"), ("compound", "similarity", "compound")],
            )

    def load(self, path):
        print("PDA Dataset: Loading previous dataset from {} ...".format(path))
        self.data = torch.load(os.path.join(self.data_folder, path))
        return self.data


def combine_datasets(original_dataset, additional_dataset, val_size, save_path="combined_dataset.pt"):
    if not original_dataset.data:
        raise ImportError("CombineDatasets: Please build the original dataset first")
    if not additional_dataset.data:
        raise ImportError("CombineDatasets: Please build the imported dataset first")

    pdbbind_drugs = additional_dataset.data["compound"].smiles
    pdbbind_proteins = additional_dataset.data["target"].names
    combined_dataset = copy.deepcopy(original_dataset)
    combined_dataset.add_compound_nodes(pdbbind_drugs)
    combined_dataset.add_target_nodes(pdbbind_proteins)
    comp_from = [
        additional_dataset.data["compound"].smiles[i]
        for i in additional_dataset.data["compound", "target"].y_edge_index[0]
    ]
    target_to = [
        additional_dataset.data["target"].names[i]
        for i in additional_dataset.data["compound", "target"].y_edge_index[1]
    ]
    edges_added = combined_dataset.add_labeled_edges(
        compound_node_names=comp_from,
        target_node_names=target_to,
        edge_labels=additional_dataset.data["compound", "target"].y.flatten().tolist(),
    )
    nb_edges_added = len(edges_added)
    total_nb_edges = combined_dataset.data["compound", "target"].y.shape[0]

    perm = torch.randperm(total_nb_edges - nb_edges_added)
    all_train_indices = torch.LongTensor([i for i in range(total_nb_edges - nb_edges_added)])
    train_size = int((1.0 - val_size) * (total_nb_edges - nb_edges_added))

    combined_dataset.data["target", "compound"].train_idx = all_train_indices[perm[:train_size]]
    combined_dataset.data["target", "compound"].val_idx = all_train_indices[perm[train_size:]]
    combined_dataset.data["target", "compound"].test_idx = torch.LongTensor([i for i in range(total_nb_edges)])[
        -nb_edges_added:
    ]
    combined_dataset.data["compound", "target"].train_idx = combined_dataset.data[
        "target", "compound"
    ].train_idx.clone()
    combined_dataset.data["compound", "target"].val_idx = combined_dataset.data["target", "compound"].val_idx.clone()
    combined_dataset.data["compound", "target"].test_idx = combined_dataset.data["target", "compound"].test_idx.clone()

    torch.save(combined_dataset.data, save_path)
    return combined_dataset.data


def export_to_dataframe(data, dataset_name):
    nb_fea_compounds, nb_fea_targets = data["compound"].x.shape[1], data["target"].x.shape[1]
    df = pd.DataFrame(
        index=range(data["target", "compound"].y.shape[0]),
        columns=["target", "compound", "set", "pAct"]
        + ["fea_target_{}".format(i) for i in range(nb_fea_targets)]
        + ["fea_compound_{}".format(i) for i in range(nb_fea_compounds)],
    )
    for idx, ((target_idx, compound_idx), pact) in enumerate(
        zip(data["target", "compound"].y_edge_index.T.tolist(), data["target", "compound"].y.flatten().tolist())
    ):
        df.loc[idx, "target"] = data["target"].names[target_idx]
        df.loc[idx, "compound"] = data["compound"].smiles[compound_idx]
        df.loc[idx, "set"] = "train" if idx in data["target", "compound"].train_idx else "test"
        df.loc[idx, "pAct"] = pact
        df.loc[idx, ["fea_target_{}".format(i) for i in range(nb_fea_targets)]] = data["target"].x[target_idx].tolist()
        df.loc[idx, ["fea_compound_{}".format(i) for i in range(nb_fea_compounds)]] = (
            data["compound"].x[compound_idx].tolist()
        )
    df.to_csv(dataset_name.replace(".pt", "_df.csv"), index=False)


def select_test_set(dataset):
    # Reduce/prune dataset to encapsulate only data from the test set
    test_edge_indices = dataset.data["target", "compound"].y_edge_index[:, dataset.data["target", "compound"].test_idx]
    test_edges_labels = dataset.data["target", "compound"].y[dataset.data["target", "compound"].test_idx]
    unique_compounds = np.unique(test_edge_indices[1, :])
    original_compounds = dataset.data["compound"].smiles
    unique_targets = np.unique(test_edge_indices[0, :])
    original_targets = dataset.data["target"].names
    mask_compcomp = np.isin(dataset.data["compound", "compound"].edge_index[0, :], unique_compounds) & np.isin(
        dataset.data["compound", "compound"].edge_index[1, :], unique_compounds
    )
    mask_tartar = np.isin(dataset.data["target", "target"].edge_index[0, :], unique_targets) & np.isin(
        dataset.data["target", "target"].edge_index[1, :], unique_targets
    )

    reduced_dataset = copy.deepcopy(dataset)
    reduced_dataset.data["compound"].smiles = np.array(reduced_dataset.data["compound"].smiles)[
        unique_compounds
    ].tolist()
    reduced_dataset.data["compound"].names = np.array(reduced_dataset.data["compound"].names)[
        unique_compounds
    ].tolist()
    reduced_dataset.data["compound"].x = reduced_dataset.data["compound"].x[unique_compounds]
    reduced_dataset.data["target"].names = np.array(reduced_dataset.data["target"].names)[unique_targets].tolist()
    reduced_dataset.data["target"].x = reduced_dataset.data["target"].x[unique_targets]
    reduced_dataset.data["compound", "compound"].edge_index = torch.LongTensor(
        [
            [
                reduced_dataset.data["compound"].smiles.index(original_compounds[i])
                for i in reduced_dataset.data["compound", "compound"].edge_index[0, mask_compcomp]
            ],
            [
                reduced_dataset.data["compound"].smiles.index(original_compounds[i])
                for i in reduced_dataset.data["compound", "compound"].edge_index[1, mask_compcomp]
            ],
        ]
    )
    reduced_dataset.data["compound", "compound"].edge_attr = reduced_dataset.data["compound", "compound"].edge_attr[
        mask_compcomp
    ]
    reduced_dataset.data["target", "target"].edge_index = torch.LongTensor(
        [
            [
                reduced_dataset.data["target"].names.index(original_targets[i])
                for i in reduced_dataset.data["target", "target"].edge_index[0, mask_tartar]
            ],
            [
                reduced_dataset.data["target"].names.index(original_targets[i])
                for i in reduced_dataset.data["target", "target"].edge_index[1, mask_tartar]
            ],
        ]
    )
    reduced_dataset.data["target", "target"].edge_attr = reduced_dataset.data["target", "target"].edge_attr[
        mask_tartar
    ]
    reduced_dataset.data["target", "compound"].y_edge_index = torch.LongTensor(
        [
            [reduced_dataset.data["target"].names.index(original_targets[i]) for i in test_edge_indices[0, :]],
            [reduced_dataset.data["compound"].smiles.index(original_compounds[i]) for i in test_edge_indices[1, :]],
        ]
    )
    reduced_dataset.data["target", "compound"].y = test_edges_labels.clone()
    reduced_dataset.data["target", "compound"].train_idx = torch.LongTensor([])
    reduced_dataset.data["target", "compound"].val_idx = torch.LongTensor([])
    reduced_dataset.data["target", "compound"].test_idx = torch.LongTensor(
        [i for i in range(test_edges_labels.shape[0])]
    )
    reduced_dataset.data["compound", "target"].y_edge_index = torch.LongTensor(
        [
            [reduced_dataset.data["compound"].smiles.index(original_compounds[i]) for i in test_edge_indices[1, :]],
            [reduced_dataset.data["target"].names.index(original_targets[i]) for i in test_edge_indices[0, :]],
        ]
    )
    reduced_dataset.data["compound", "target"].y = test_edges_labels.clone()
    reduced_dataset.data["compound", "target"].train_idx = reduced_dataset.data["target", "compound"].train_idx.clone()
    reduced_dataset.data["compound", "target"].val_idx = reduced_dataset.data["target", "compound"].val_idx.clone()
    reduced_dataset.data["compound", "target"].test_idx = reduced_dataset.data["target", "compound"].test_idx.clone()

    return reduced_dataset
