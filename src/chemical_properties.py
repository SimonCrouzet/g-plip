from rdkit import Chem, DataStructs
from rdkit.Chem.Descriptors import ExactMolWt, TPSA
from rdkit.Chem.rdMolDescriptors import (
    CalcNumRotatableBonds,
    CalcNumLipinskiHBA,
    CalcNumLipinskiHBD,
    CalcNumAromaticRings,
    CalcNumAliphaticRings,
)
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, GetHashedMorganFingerprint
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
import numpy as np
import logging


def get_smiles_from_mol2(mol2_file):
    try:
        # Try to process MOL2 file
        mol = Chem.MolFromMol2File(mol2_file)
        # If compound was correctly computed, save the SMILES representation
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles
    except:
        # If it was impossible to build molecule or retrieve SMILES representation due to a wrong sdf/mol2 file, just skip the sample
        logging.error("Impossible to build molecule or retrieve SMILES representation from file {}".format(mol2_file))
        return None


"""
For the compounds, check if the SMILES representation from the PACE library is not miscallaneous, otherwise exclude the compound.
To check miscallaneous representation, we are trying to compute compound features
"""


def check_compounds_smiles_from_PACE(pace_res_df, compounds):
    excluded_compounds = []
    updated_comp2smiles = {}
    for comp in compounds:
        comp_smiles = pace_res_df[pace_res_df.CORE == comp].SMILES_standard.unique()
        if len(comp_smiles) == 0:  # No matching SMILES for the compound
            excluded_compounds.append(comp)
        elif len(comp_smiles) > 1:  # Several matching SMILES for the compound
            # Check parent isotopes
            parent_isotopes = []
            for isotope in comp_smiles:
                parent_isotopes.append(
                    Chem.MolToSmiles(Chem.MolStandardize.rdMolStandardize.IsotopeParent(Chem.MolFromSmiles(isotope)))
                )

            if len(np.unique(parent_isotopes)) == 1:  # One single parent isotope
                pace_res_df[pace_res_df.CORE == comp].SMILES_standard.replace(
                    {isotope: parent_isotopes[0] for isotope in comp_smiles}
                )
                if check_compounds_features(parent_isotopes[0]):
                    updated_comp2smiles[comp] = parent_isotopes[0]
                else:
                    excluded_compounds.append(comp)
            else:  # Several parent isotopes
                excluded_compounds.append(comp)
                logging.warning(
                    "Compound {} has several SMILES representation for several parent isotopes!".format(comp)
                )
        else:
            if check_compounds_features(comp_smiles[0]):
                updated_comp2smiles[comp] = comp_smiles[0]
            else:
                excluded_compounds.append(comp)
    logging.warning(
        "Dataset: The following compounds were excluded from the pipeline as lacking or confusing SMILES representation from the PACE library:\n{}\n".format(
            excluded_compounds
        )
    )
    return [c for c in compounds if c not in excluded_compounds], updated_comp2smiles


"""
For the compounds, check if the SMILES representation from the screening file is not miscallaneous, otherwise exclude the compound.
To check miscallaneous representation, we are trying to compute compound features
"""


def check_compounds_smiles(compound2smiles, compounds):
    excluded_compounds = []
    updated_comp2smiles = {}
    for comp, comp_smiles in compound2smiles.items():
        if len(comp_smiles) == 0:
            excluded_compounds.append(comp)
        elif len(comp_smiles) > 1:  # Several different SMILES for the same compound
            parent_isotopes = []
            for isotope in comp_smiles:
                parent_isotopes.append(
                    Chem.MolToSmiles(Chem.MolStandardize.rdMolStandardize.IsotopeParent(Chem.MolFromSmiles(isotope)))
                )

            if len(np.unique(parent_isotopes)) == 1:  # One single parent isotope
                if check_compounds_features(parent_isotopes[0]):
                    updated_comp2smiles[comp] = parent_isotopes[0]
                else:
                    excluded_compounds.append(comp)
                    # logging.warning('PDA Dataset: Compound {} has a common parent isotope thus with a misleading SMILES representation (chemical features unable to be extracted with RDKit)!'.format(comp)
            else:  # Several parent isotopes
                excluded_compounds.append(comp)
                logging.warning(
                    "Compound {} has several SMILES representation for several parent isotopes!".format(comp)
                )
                # raise ValueError('PDA Dataset: Compound {} have several SMILES representation! {}'.format(comp, comp_smiles))
        else:
            if check_compounds_features(comp_smiles[0]):
                updated_comp2smiles[comp] = comp_smiles[0]
            else:
                excluded_compounds.append(comp)
                # logging.warning('PDA Dataset: Compound {} has a misleading SMILES representation (chemical features unable to be extracted with RDKit)!'.format(comp))
    logging.warning(
        "Dataset: The following compounds were excluded from the pipeline as lacking or confusing SMILES representation from the PACE library:\n{}\n".format(
            excluded_compounds
        )
    )
    return [c for c in compounds if c not in excluded_compounds], updated_comp2smiles


def check_compounds_features(smile):
    # Check if compound features are correctly computed
    return get_compound_features(smile) is not None


def get_compound_smiles(compound, pace_res_df):
    # We assume the unicity of SMILES for a compound - checked before to reduce computational time
    return pace_res_df[pace_res_df.CORE == compound].SMILES_standard.unique()[0]


def get_compound_features(
    compound_smiles,
    to_list=False,
    fingerprint="ECFP",
    fingerprint_size=124,
    lipinski_features=True,
    other_features=True,
):
    try:
        compound_features = {}
        compound_molecule = Chem.MolFromSmiles(compound_smiles)

        if fingerprint is None and not lipinski_features and not other_features:
            # No compound features, return 'Dummy' features
            return [1.0]

        if fingerprint_size is None:
            fingerprint_size = 2048  # Default value proposed by rdKit

        if lipinski_features:
            compound_features["molecular_weight"] = ExactMolWt(compound_molecule)
            compound_features["nb_h_donors"] = CalcNumLipinskiHBD(compound_molecule)
            compound_features["nb_h_acceptors"] = CalcNumLipinskiHBA(compound_molecule)
            compound_features["octanol_water_partition_coef"] = MolLogP(
                compound_molecule
            )  # Used as a measure of lipophilicity

        if other_features:
            compound_features["tpsa"] = TPSA(compound_molecule)  # Topological Polar Surface Area
            compound_features["nb_rot_bonds"] = CalcNumRotatableBonds(compound_molecule)
            compound_features["molecular_refractivity"] = MolMR(compound_molecule)
            compound_features["nb_atoms"] = compound_molecule.GetNumAtoms()
            compound_features["nb_arom_rings"] = CalcNumAromaticRings(compound_molecule)
            compound_features["nb_aliph_rings"] = CalcNumAliphaticRings(compound_molecule)
            # Evaluate aromaticity - ugly method
            compound_features["percentage_aromaticity"] = (
                sum(
                    [
                        compound_molecule.GetAtomWithIdx(i).GetIsAromatic()
                        for i in range(compound_molecule.GetNumAtoms())
                    ]
                )
                / compound_molecule.GetNumAtoms()
            )

        if fingerprint == "ECFP":
            # Encoded in 248 bits as adviced by Leo to not overflow the neurons
            fp = GetMorganFingerprintAsBitVect(
                compound_molecule, 2, nBits=fingerprint_size
            )  # Radius 2 so roughly equals to ECFP4
            temp_array = np.zeros((0,), dtype=np.int8)
        elif fingerprint == "FCFP":
            fp = GetMorganFingerprintAsBitVect(
                compound_molecule, 2, useFeatures=True, nBits=fingerprint_size
            )  # Radius 2 so roughly equals to FCFP4
            temp_array = np.zeros((0,), dtype=np.int8)
        elif fingerprint == "ECFP_Count":
            fp = GetHashedMorganFingerprint(compound_molecule, 2, nBits=fingerprint_size)
            temp_array = np.zeros((0,), dtype=np.int32)
        elif fingerprint == "FCFP_Count":
            fp = GetHashedMorganFingerprint(compound_molecule, 2, nBits=fingerprint_size)
            temp_array = np.zeros((0,), dtype=np.int32)
        elif fingerprint is None or fingerprint == "None":
            compound_fingerprint = []
        else:
            raise ValueError("Compound Features: Fingerprint {} not implemented!".format(fingerprint))

        if fingerprint is not None and fingerprint != "None":
            # Convert fingerprint to a correct data structure
            DataStructs.ConvertToNumpyArray(fp, temp_array)
            compound_fingerprint = temp_array.tolist()

        if to_list:
            return list(compound_features.values()) + compound_fingerprint
        else:
            compound_features["fingerprint"] = compound_fingerprint
            return compound_features
    except Exception as e:
        print(
            "Chemical features: unable to compute for compound {} due to the following error: {}".format(
                compound_smiles, e
            )
        )
        logging.warning(
            "Compound with uncorrect SMILES: {}. It leads to the following error: {}".format(compound_smiles, e)
        )
        return None


def get_nb_compound_features(fingerprint="ECFP", fingerprint_size=124, lipinski_features=True, other_features=True):
    # Check number of compound features with a basic example
    return len(
        get_compound_features(
            "C",
            to_list=True,
            fingerprint=fingerprint,
            fingerprint_size=fingerprint_size,
            lipinski_features=lipinski_features,
            other_features=other_features,
        )
    )


def compounds_similarity(compound1, compound2):
    # Calculate compound similarity
    compound_molecule1 = Chem.MolFromSmiles(compound1)
    compound_molecule2 = Chem.MolFromSmiles(compound2)
    fp1 = Chem.RDKFingerprint(compound_molecule1)
    fp2 = Chem.RDKFingerprint(compound_molecule2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compare_compounds(compound1, compound2):
    # Check if two compounds are the same
    return compounds_similarity(compound1, compound2) == 1.0


def get_scaffolds(comp2smiles):
    # From SMILES representation, get the different scaffolds
    comp2scaffold = {}
    scaffolds = set()

    for comp, smiles in comp2smiles.items():
        scaff = Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles(smiles)))
        comp2scaffold[comp] = scaff
        scaffolds.add(scaff)
    scaffold_idx = {i: s for i, s in enumerate(scaffolds)}

    return comp2scaffold, scaffold_idx


def check_Lipinski_violations(compound_smiles):
    # Check how many Lipinski (RO5) rules the compound violates
    ro5_violation_score = 0

    compound_molecule = Chem.MolFromSmiles(compound_smiles)

    if ExactMolWt(compound_molecule) > 500:
        ro5_violation_score += 1
    if CalcNumLipinskiHBD(compound_molecule) > 5:
        ro5_violation_score += 1
    if CalcNumLipinskiHBA(compound_molecule) > 10:
        ro5_violation_score += 1
    if MolLogP(compound_molecule) > 5:  # Used as a measure of lipophilicity
        ro5_violation_score += 1
    return ro5_violation_score


def get_intersection_smiles(smiles_list1, smiles_list2):
    # Get the intersection of two lists of SMILES - here we carefully check SMILES one by one to avoid similar molecules with different SMILES to be considered as different
    # /!\ Slow method, use it only to be sure that the intersection is correct
    intersection_smiles = []
    for smiles1 in smiles_list1:
        for smiles2 in smiles_list2:
            try:
                if compare_compounds(smiles1, smiles2):
                    intersection_smiles.append(smiles1)
                    break
            except:
                continue
    return intersection_smiles


def get_union_smiles(smiles_list1, smiles_list2):
    # Get the union of two lists of SMILES - here we carefully check SMILES one by one to avoid similar molecules with different SMILES to be added twice
    # /!\ Slow method, use it only to be sure that the union is correct
    union_smiles = []
    idx_list1 = []
    for idx, smiles in enumerate(smiles_list1):
        found = False
        for registered_smiles in union_smiles:
            try:
                if compare_compounds(smiles, registered_smiles):
                    found = True
                    break
            except:
                continue
        if not found:
            try:
                # Still check if the SMILES is not miscalleneous
                _ = Chem.MolFromSmiles(smiles)
                union_smiles.append(smiles)
                idx_list1.append(idx)
            except:
                continue

    idx_list2 = []
    for idx, smiles in enumerate(smiles_list2):
        found = False
        for registered_smiles in union_smiles:
            try:
                if compare_compounds(smiles, registered_smiles):
                    found = True
                    break
            except:
                continue
        if not found:
            try:
                # Still check if the SMILES is not miscalleneous
                _ = Chem.MolFromSmiles(smiles)
                union_smiles.append(smiles)
                idx_list2.append(idx)
            except:
                continue
    return union_smiles, (idx_list1, idx_list2)


def find_smiles_index(smiles, smiles_list):
    # Find the index of a SMILES in a list of SMILES - here we carefully check SMILES one by one to avoid similar molecules with different SMILES to be considered as different
    # /!\ Slow method, use it only to be sure that the index is correct
    for idx, smiles_in_list in enumerate(smiles_list):
        try:
            if compare_compounds(smiles, smiles_in_list):
                return idx
        except:
            continue
    return None
