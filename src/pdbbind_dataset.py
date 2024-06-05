import itertools as it
import json
import logging
import os
import random
import sys
from statistics import fmean

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from tqdm.auto import tqdm

from src.chemical_properties import compare_compounds, get_smiles_from_mol2
from src.data import (
    HeteroData,
    PDADataset,
    compounds_similarity,
    dataframe_raw_vectorization_with_numpy,
    generate_esm_encodings,
    get_compound_features,
    get_nb_compound_features,
    get_nb_protein_features,
    get_protein_features,
    stratified_random_train_set_split,
)
from src.external import download_uniprot_fasta, exclude_nonhuman_proteins

"""
Class adapted from the PDADataset class to load PDB Bind data
TODO: Refactor PDA Dataset to avoid such copy/paste!
"""


class PDBBindDataset(PDADataset):
    def __init__(self, settings, set="complete"):
        # Specify which part of the PDB Bind dataset to build
        if set == "complete":  # Complete set
            self.pdb_set = "complete"
        elif set == "refined":  # Refined set - curated high quality complexes
            self.pdb_set = "refined"
        elif set == "other":  # Other - all other complexes
            self.pdb_set = "other"
        else:
            raise ValueError("Unknown PDB set {} - can only be 'complete', 'refined' or 'other'".format(set))

        super().__init__(settings)

    def compound_dict(self, pdbbind_all_index):
        smiles_to_equivalent = {}
        invalid_smiles = []
        for smile in tqdm(pdbbind_all_index["SMILES"].unique()):
            try:
                exclusive = True
                for key, value_list in smiles_to_equivalent.items():
                    if compare_compounds(smile, key):
                        value_list.append(smile)
                        exclusive = False
                        break
                if exclusive:
                    smiles_to_equivalent[smile] = [smile]
            except Exception:
                invalid_smiles.append(smile)

        # TODO: Just realized that values -> keys makes more sense - fix that properly!
        new_dict = {}
        for k in smiles_to_equivalent.keys():
            for v in smiles_to_equivalent[k]:
                new_dict[v] = k
        return new_dict, invalid_smiles

    @property
    def processed_file_names(self):
        return super().processed_file_names.replace("pda-data", "pdbbind-data-{}".format(self.pdb_set))

    def read_data(self):
        pdbbind_data_folder = os.path.join(self.data_folder, "v2020-{}-PL".format(self.pdb_set))
        try:
            ppi_df = pd.read_csv(
                os.path.join(self.data_folder, self.settings.INPUT_FILES.ppi_database),
                header=None,
                names=["Prot1", "Prot2"],
            )
            gene_expr_df = pd.read_csv(
                os.path.join(self.data_folder, self.settings.INPUT_FILES.gene_expr_database), delimiter="\t"
            )
            pdbbind_general_df = pd.read_csv(
                os.path.join(pdbbind_data_folder, "index", "INDEX_general_PL_data.2020"),
                delimiter=r"\s{2,}",
                engine="python",
                skiprows=6,
                header=None,
                on_bad_lines="skip",
            )
            pdbbind_general_df = pdbbind_general_df.drop(columns=[4, 5]).rename(
                columns={0: "PDB code", 1: "resolution", 2: "release year", 3: "-logKd/Ki"}
            )
            pdbbind_name_df = pd.read_csv(
                os.path.join(pdbbind_data_folder, "index", "INDEX_general_PL_name.2020"),
                delimiter=r"\s{2,}",
                engine="python",
                skiprows=6,
                header=None,
                on_bad_lines="skip",
            )
            pdbbind_name_df = pdbbind_name_df.drop(columns=[1, 3]).rename(columns={0: "PDB code", 2: "UniProt ID"})
        except Exception:
            raise ValueError(
                "PDB Bind data not found in {} - please fix your folder containing PDB data accordingly".format(
                    pdbbind_data_folder
                )
            )

        pdbbind_all_index = pdbbind_general_df.merge(pdbbind_name_df, on="PDB code")

        SMILES_dict = {}
        for pdb_code in pdbbind_all_index["PDB code"]:
            # Several issues with sdf and mol2 files from the PDB Bind dataset
            try:
                # Try to process MOL2 file
                mol = Chem.MolFromMol2File(
                    os.path.join(pdbbind_data_folder, pdb_code, "{}_ligand.mol2".format(pdb_code))
                )
                if mol is None:
                    # If MOL2 file didn't work, try with SDF file
                    mol = Chem.SDMolSupplier(
                        os.path.join(pdbbind_data_folder, pdb_code, "{}_ligand.sdf".format(pdb_code))
                    )[0]
                # If compound was correctly computed, save the SMILES representation
                SMILES_dict[pdb_code] = Chem.MolToSmiles(mol, isomericSmiles=True)
            except Exception:
                # If it was impossible to build molecule or retrieve SMILES representation due to a wrong sdf/mol2
                # file, just skip the sample
                continue

        smiles_temp = pd.DataFrame({"PDB code": list(SMILES_dict.keys()), "SMILES": list(SMILES_dict.values())})

        pdbbind_all_index = pdbbind_all_index.merge(smiles_temp, on="PDB code")

        self.targets = pdbbind_all_index["UniProt ID"].unique()
        self.targets = exclude_nonhuman_proteins(
            self.targets, json_path=os.path.join(self.data_folder, "pdbbind_human_proteins.json")
        )

        mapping_file = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.mapping_file), delimiter="\t"
        )
        mapping_file = mapping_file[mapping_file.Accession.isin(self.targets)]
        self.targets = mapping_file.Accession.unique().tolist()

        pdbbind_all_index = pdbbind_all_index[pdbbind_all_index["UniProt ID"].isin(self.targets)]

        self.compounds = pdbbind_all_index["SMILES"].unique().tolist()  # Provide SMILES directly here

        # Save correct SMILES and exclude miscallaneous ones (which were not suitable to calculate compound features)
        smiles_eq_path = os.path.join(self.data_folder, "pdbbind_smiles_mapping.json")
        if os.path.isfile(smiles_eq_path):
            with open(smiles_eq_path) as json_file:
                smiles_to_equivalent_save = json.load(json_file)
                smiles_to_equivalent, invalid_smiles = (
                    smiles_to_equivalent_save["smiles_dict"],
                    smiles_to_equivalent_save["invalid_smiles"],
                )
        else:
            smiles_to_equivalent, invalid_smiles = self.compound_dict(pdbbind_all_index)
            smiles_to_equivalent_save = {"smiles_dict": smiles_to_equivalent, "invalid_smiles": invalid_smiles}
            with open(smiles_eq_path, "w+") as json_file:
                json.dump(smiles_to_equivalent_save, json_file)
        self.compounds = np.unique(
            [v for k, v in smiles_to_equivalent.items() if k in pdbbind_all_index["SMILES"].tolist()]
        ).tolist()
        pdbbind_all_index = pdbbind_all_index[pdbbind_all_index["SMILES"].isin(self.compounds)]

        download_uniprot_fasta(self.targets, self.data_folder)

        self.pdbbind_all_index = pdbbind_all_index.reset_index(drop=True)

        return pdbbind_all_index, ppi_df, gene_expr_df, mapping_file, smiles_to_equivalent

    def build_graph(self):
        pdbbind_all_index, ppi_df, gene_expr_df, mapping_file, smiles_to_equivalent = self.read_data()

        print(
            "Initiating a graph containing {} compounds and {} targets...".format(
                len(self.compounds), len(self.targets)
            )
        )

        # Protein features
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
        for idx, target in enumerate(tqdm(self.targets, file=sys.stdout, desc="Protein features")):
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
            )

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
        for idx, compound_smiles in enumerate(tqdm(self.compounds, file=sys.stdout, desc="Compound features")):
            compound_features[idx, :] = get_compound_features(
                smiles_to_equivalent[compound_smiles],
                to_list=True,
                fingerprint=fingerprint,
                fingerprint_size=self.settings.COMPOUND_FEATURES.fingerprints_size,
                lipinski_features=self.settings.COMPOUND_FEATURES.lipinski_features,
                other_features=self.settings.COMPOUND_FEATURES.extended_chemical_features,
            )

        # Edges for compound-targets interactions
        edges_inter_from_to = []
        # edges_inter_features = []
        edges_inter_y = []
        edges_inter_train_idx, edges_inter_test_idx = [], []

        # Create a dict of experimental pAct
        edge_inter_dict = {prot: {compound: [] for compound in self.compounds} for prot in self.targets}
        edge_inter_dict_holdout = {prot: {compound: [] for compound in self.compounds} for prot in self.targets}
        # Get pAct info
        for target_uniprot_id, compound_smiles_name, pAct, year in tqdm(
            dataframe_raw_vectorization_with_numpy(
                pdbbind_all_index, keys=["UniProt ID", "SMILES", "-logKd/Ki", "release year"]
            ),
            file=sys.stdout,
            desc="Edges pAct",
        ):
            if compound_smiles_name in self.compounds and target_uniprot_id in self.targets:
                if year < 2019:
                    edge_inter_dict[target_uniprot_id][compound_smiles_name].append(pAct)
                else:
                    edge_inter_dict_holdout[target_uniprot_id][compound_smiles_name].append(pAct)
            elif compound_smiles_name in smiles_to_equivalent and target_uniprot_id in self.targets:
                if smiles_to_equivalent[compound_smiles_name] not in self.compounds:
                    logging.warning(f"{smiles_to_equivalent[compound_smiles_name]} not in self.compounds")
                if year < 2019:
                    edge_inter_dict[target_uniprot_id][smiles_to_equivalent[compound_smiles_name]].append(pAct)
                else:
                    edge_inter_dict_holdout[target_uniprot_id][smiles_to_equivalent[compound_smiles_name]].append(pAct)
            else:
                logging.warning(
                    "PDADataset: pAct recorded on unknown drug or target (compound {} - target {})".format(
                        compound_smiles_name, target_uniprot_id
                    )
                )

        idx_count = 0
        for uniprot_id in self.targets:
            for compound_smiles_name in self.compounds:
                # PDB Bind 2016 data - Used for train set
                if edge_inter_dict[uniprot_id][compound_smiles_name]:  # list is not empty
                    edges_inter_from_to.append([compound_smiles_name, uniprot_id])
                    edges_inter_train_idx.append(idx_count)
                    idx_count += 1
                    if self.settings.INPUT_FILES.average_screenings:
                        # Average and fill in edges
                        edges_inter_y.append([fmean(edge_inter_dict[uniprot_id][compound_smiles_name])])
                    else:
                        # Pick randomly one of the pAct and fill in edges
                        edges_inter_y.append([random.choice(edge_inter_dict[uniprot_id][compound_smiles_name])])
                # PDB Bind 2020 data - Used for test set
                if edge_inter_dict_holdout[uniprot_id][compound_smiles_name]:  # list is not empty
                    edges_inter_from_to.append([compound_smiles_name, uniprot_id])
                    edges_inter_test_idx.append(idx_count)
                    idx_count += 1
                    if self.settings.INPUT_FILES.average_screenings:
                        # Average and fill in edges
                        edges_inter_y.append([fmean(edge_inter_dict_holdout[uniprot_id][compound_smiles_name])])
                    else:
                        # Pick randomly one of the pAct and fill in edges
                        edges_inter_y.append(
                            [random.choice(edge_inter_dict_holdout[uniprot_id][compound_smiles_name])]
                        )

        # Edges for protein-protein interactions
        edges_ppi_from_to = []
        edges_ppi_features = []

        for prot1_geneid, prot2_geneid in tqdm(
            dataframe_raw_vectorization_with_numpy(ppi_df, keys=["Prot1", "Prot2"]), file=sys.stdout, desc="PPI edges"
        ):
            # We are using a PPI network referencing gene IDs -> Translate them to UniProt IDs
            # (there can be several proteins!)
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
        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            similarity_threshold = self.get_compound_similarity_threshold()

            edges_drugs_from_to = []
            edges_drugs_features = []

            for comp1, comp2 in tqdm(
                it.combinations(self.compounds, r=2), file=sys.stdout, desc="Compound similarity edges"
            ):
                if self.settings.COMPOUND_FEATURES.compound_similarities:
                    # Use compound similarities on the edges
                    # edges_drugs_features.append(compounds_similarity(get_compound_smiles(comp1, pace_res_df),
                    # get_compound_smiles(comp2, pace_res_df)))
                    similarity = compounds_similarity(smiles_to_equivalent[comp1], smiles_to_equivalent[comp2])
                    if similarity >= similarity_threshold:
                        edges_drugs_from_to.append([comp1, comp2])
                        edges_drugs_features.append(similarity)
                else:
                    # Use basic 1-to-1 links
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
        # edges_inter_features = np.array(edges_inter_features)
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

        data["target"].x = torch.Tensor(target_features)
        data["target"].names = self.targets
        data["compound"].x = torch.Tensor(compound_features)
        data["compound"].names = self.compounds
        data["compound"].smiles = [smiles_to_equivalent[k] for k in self.compounds]

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

        # Set label to predict as pAct for protein-drug interaction
        data["compound", "interaction", "target"].edge_index = torch.ones(
            (2, 0)
        ).long()  # Create empty edge index - for compatibility
        data["compound", "interaction", "target"].edge_attr = torch.ones(
            (0, 1)
        ).float()  # Create empty edge attr - for compatibility
        data["compound", "interaction", "target"].y_edge_index = torch.LongTensor(edges_inter_from_to)
        data["compound", "interaction", "target"].y = torch.Tensor(edges_inter_y)
        data["compound", "interaction", "target"].train_idx = torch.LongTensor(edges_inter_train_idx)
        data["compound", "interaction", "target"].val_idx = torch.ones((0)).long()
        data["compound", "interaction", "target"].test_idx = torch.LongTensor(edges_inter_test_idx)

        data = self.transform_graph(data)

        print("Graph initialized!\n{}\n\n".format(data))
        return data

    def get(self):
        if os.path.isfile(os.path.join(self.data_folder, self.processed_file_names)):
            print(
                "PDBBind Dataset: Loading data from {} ...".format(
                    os.path.join(self.data_folder, self.processed_file_names)
                )
            )
            self.data = torch.load(os.path.join(self.data_folder, self.processed_file_names))
            _ = self.read_data()
        else:
            print("PDBBind Dataset: Building data...")
            self.data = self.build_process()
        return self.data

    def export_data(self, data, only_test=False, file_name=None):
        if file_name is None:
            file_name = self.processed_file_names
        # Export data to csv file
        if only_test:
            test_idx = data["compound", "target"].test_idx
            test_compounds = [data["compound"].smiles[i] for i in data["compound", "target"].y_edge_index[0][test_idx]]
            test_targets = [data["target"].names[i] for i in data["compound", "target"].y_edge_index[1][test_idx]]
            test_pAct = data["compound", "target"].y[test_idx].flatten().tolist()
            df = pd.DataFrame(data={"compound": test_compounds, "target": test_targets, "pAct": test_pAct})
            df.to_csv(os.path.join(self.data_folder, file_name.replace(".pt", "_test.csv")), index=False)
        else:
            all_compounds = [data["compound"].smiles[i] for i in data["compound", "target"].y_edge_index[0]]
            all_targets = [data["target"].names[i] for i in data["compound", "target"].y_edge_index[1]]
            all_pAct = data["compound", "target"].y.flatten().tolist()
            all_sets = [
                (
                    "train"
                    if i in data["compound", "target"].train_idx
                    else "val" if i in data["compound", "target"].val_idx else "test"
                )
                for i in range(len(all_pAct))
            ]
            df = pd.DataFrame(
                data={"compound": all_compounds, "target": all_targets, "pAct": all_pAct, "set": all_sets}
            )
            df.to_csv(os.path.join(self.data_folder, file_name.replace(".pt", ".csv")), index=False)

    def transform_graph(self, data):
        data = super().transform_graph(data)

        data["target", "compound"].train_idx = torch.LongTensor(data["compound", "target"].train_idx.tolist())
        data["target", "compound"].val_idx = torch.LongTensor(data["compound", "target"].val_idx.tolist())
        data["target", "compound"].test_idx = torch.LongTensor(data["compound", "target"].test_idx.tolist())
        return data


def construct_pdbbind_data(params, output_f, pdbbind_set="complete"):
    dataset = PDBBindDataset(settings=params, set=pdbbind_set)

    data = dataset.get()

    torch.manual_seed(params.RUNPARAMS.seed)
    # train/test split is directly built in PDBBind - we rely on the v2019 temporal split

    # Thus, we still need to build the validation set:
    if params.RUNPARAMS.val_size > 0.0:
        data = stratified_random_train_set_split(
            data,
            edge_type=params.MODELINFO.link_prediction_mode,
            val_perc=params.RUNPARAMS.val_size,
            test_perc=params.RUNPARAMS.test_size,
        )
        data[list(reversed(params.MODELINFO.link_prediction_mode))].train_idx = data[
            params.MODELINFO.link_prediction_mode
        ].train_idx.clone()
        data[list(reversed(params.MODELINFO.link_prediction_mode))].val_idx = data[
            params.MODELINFO.link_prediction_mode
        ].val_idx.clone()
        data[list(reversed(params.MODELINFO.link_prediction_mode))].test_idx = data[
            params.MODELINFO.link_prediction_mode
        ].test_idx.clone()

    output_f.flush()

    return data, dataset


def construct_pdbbind_core_data(params, output_f, pdbbind_set="complete"):
    dataset = PDBBindDataset(settings=params, set=pdbbind_set)

    data = dataset.get()

    torch.manual_seed(params.RUNPARAMS.seed)

    casf_filenames = dataset.processed_file_names.replace(".pt", "_casf.pt")
    print("CASF file name: {}".format(casf_filenames))
    if os.path.isfile(os.path.join(dataset.data_folder, casf_filenames)):
        data = torch.load(os.path.join(dataset.data_folder, casf_filenames))
        # /!\: Careful about dataset.compounds and dataset.targets - they are not updated
    else:
        dataset = casf_core_split(dataset, params)
        torch.save(dataset.data, os.path.join(dataset.data_folder, casf_filenames))
        data = dataset.data

    output_f.flush()

    # return data, dataset
    return data, dataset


def casf_core_split(
    dataset, params, casf_subfolder="CASF-2016", casf_mapping_file="CASF_Mapping_HumanProteins_20230627.tsv"
):
    casf_folder = os.path.join(params.RUNINFO.data_dir, casf_subfolder)
    core_df = pd.read_csv(os.path.join(casf_folder, "power_screening", "CoreSet.dat"), sep="\s+")
    ligand_df = pd.read_csv(os.path.join(casf_folder, "power_screening", "LigandInfo.dat"), skiprows=8, sep="\s+")

    # pdb_mapping_df = PDBCode_to_UniProtID(target_df['#T'].values)
    # UniProt ID Mapping API is dysfunctional - manually requested on the website
    pdb_mapping_df = pd.read_csv(os.path.join("data", casf_mapping_file), sep="\t")
    pdb_mapping_df = pdb_mapping_df.drop(["Organism", "Entry Name"], axis=1).rename(
        columns={"Entry": "UniProtID", "From": "PDBCode"}
    )

    pdb2uniprot = {}
    for pdbcode, uniprotid in zip(pdb_mapping_df["PDBCode"], pdb_mapping_df["UniProtID"]):
        if pdbcode in pdb2uniprot:
            # Manual fix - uniprot has multiple entries for the same PDB code, we then checked on RCSB
            if pdbcode == "4w9h":
                pdb2uniprot[pdbcode] = "Q15370"
            elif pdbcode == "3utu":
                pdb2uniprot[pdbcode] = "P00734"
            elif pdbcode == "2p15":
                pdb2uniprot[pdbcode] = "P03372"
            else:
                print(
                    "PDB code {} is mapped to multiple UniProt IDs: {} and {}".format(
                        pdbcode, pdb2uniprot[pdbcode], uniprotid
                    )
                )
        else:
            pdb2uniprot[pdbcode] = uniprotid

    target_ligand_pair = []
    for pdbcode, t1, t2, target_id in zip(ligand_df["#code"], ligand_df["T1"], ligand_df["T2"], ligand_df["group"]):
        if t1 not in pdb2uniprot.keys():
            # Some proteins don't have UniProt ID or are not from Human - skipped
            continue
        if os.path.isfile(os.path.join(casf_folder, "coreset", pdbcode, pdbcode + "_ligand_opt.mol2")):
            ligand_smiles = get_smiles_from_mol2(
                os.path.join(casf_folder, "coreset", pdbcode, pdbcode + "_ligand_opt.mol2")
            )
        elif os.path.isfile(os.path.join(casf_folder, "coreset", pdbcode, pdbcode + "_ligand.mol2")):
            ligand_smiles = get_smiles_from_mol2(
                os.path.join(casf_folder, "coreset", pdbcode, pdbcode + "_ligand.mol2")
            )
        else:
            raise ValueError("Ligand file not found for PDB code {}".format(pdbcode))
        if ligand_smiles is not None:
            target_ligand_pair.append((pdb2uniprot[t1], ligand_smiles, pdbcode))
            if t2 is not np.NaN and t2 in pdb2uniprot.keys():
                target_ligand_pair.append((pdb2uniprot[t2], ligand_smiles, pdbcode))

    target_ligand_labels = []
    for target, ligand, pdbcode in target_ligand_pair:
        target_ligand_labels.append(core_df[(core_df["#code"] == pdbcode)]["logKa"].item())

    unique_targets = set([x[0] for x in target_ligand_pair])
    unique_ligands = set([x[1] for x in target_ligand_pair])

    # It was already excluded when downloading the mapping data
    # all_human_proteins = exclude_nonhuman_proteins(list(unique_targets), json_path=os.path.join(self.data_folder,
    # 'pdbbind_human_proteins.json'))
    # unique_targets = [x for x in unique_targets if x in all_human_proteins]
    # target_ligand_pair = [(x,y) for x,y in target_ligand_pair if x in unique_targets]
    # unique_ligands = list(set([x[1] for x in target_ligand_pair]))

    dataset.add_target_nodes(list(unique_targets))
    dataset.add_compound_nodes(list(unique_ligands))
    dataset.add_labeled_edges(
        compound_node_names=[x[1] for x in target_ligand_pair],
        target_node_names=[x[0] for x in target_ligand_pair],
        edge_labels=target_ligand_labels,
    )

    core_all_index = pd.DataFrame(target_ligand_pair, columns=['UniProt ID', 'SMILES', 'PDB code'])
    core_all_index['-logKd/Ki'] = target_ligand_labels
    core_all_index['release year'] = np.nan
    dataset.pdbbind_all_index = pd.concat([dataset.pdbbind_all_index, core_all_index], ignore_index=True).reset_index(drop=True)

    existing_edges = [(t.item(), l.item()) for t, l in dataset.data["target", "compound"].y_edge_index.T]
    casf_edges = [
        (dataset.data["target"].names.index(x[0]), dataset.data["compound"].names.index(x[1]))
        for x in target_ligand_pair
    ]

    all_train_edges = torch.LongTensor(
        [edge_idx for edge_idx, edge in enumerate(existing_edges) if edge not in casf_edges]
    )
    test_edges = torch.LongTensor([edge_idx for edge_idx, edge in enumerate(existing_edges) if edge in casf_edges])

    perm = torch.randperm(all_train_edges.shape[0])
    train_size = int((1.0 - params.RUNPARAMS.val_size) * all_train_edges.shape[0])

    dataset.data["target", "compound"].train_idx = all_train_edges[perm[:train_size]]
    dataset.data["target", "compound"].val_idx = all_train_edges[perm[train_size:]]
    dataset.data["target", "compound"].test_idx = test_edges
    dataset.data["compound", "target"].train_idx = dataset.data["target", "compound"].train_idx.clone()
    dataset.data["compound", "target"].val_idx = dataset.data["target", "compound"].val_idx.clone()
    dataset.data["compound", "target"].test_idx = dataset.data["target", "compound"].test_idx.clone()

    casf_processed_file_names = dataset.processed_file_names.replace(".pt", "_casf.pt")
    torch.save(dataset.data, os.path.join(dataset.data_folder, casf_processed_file_names))
    dataset.export_data(dataset.data, only_test=True, file_name=casf_processed_file_names)
    dataset.export_data(dataset.data, only_test=False, file_name=casf_processed_file_names)

    return dataset
