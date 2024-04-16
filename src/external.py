import json
import logging
import os
import subprocess
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from src.utils import dump_to_json

# Default settings to have correct requests to external servors (no DDOS)
POLLING_INTERVAL = 5
API_URL = "https://rest.uniprot.org"
retries = Retry(total=10, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

# Sublocation dictionary
sublocations_dict = [
    "Autophagosome",
    "Caveola",
    "Cell cortex",
    "Cell membrane",
    "Cell projection",
    "Cell surface",
    "Chromaffin granule",
    "Coated pit",
    "Contractile vacuole",
    "COPI-coated vesicle",
    "COPII-coated vesicle",
    "Cytoplasm",
    "Cytoplasmic granule",
    "Cytoskeleton",
    "Endoplasmic reticulum",
    "Endoplasmic reticulum-Golgi intermediate compartment",
    "Endosome",
    "Focal adhesion",
    "Golgi apparatus",
    "Hydrogenosome",
    "Lipid droplet",
    "Lysosome",
    "Melanosome",
    "Mitochondrion",
    "Nucleus",
    "Organellar chromatophore",
    "Perinuclear region",
    "Peroxisome",
    "Phagocytic cup",
    "Phagosome",
    "Secreted",
]


# Retrieve sublocations from UniProt query
def request_sublocation_from_uniprot(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=accession:{uniprot_id}"
    +"+AND+organism_id:9606&format=tsv&fields=cc_subcellular_location"
    df = pd.read_csv(url, delimiter="\t")
    if len(df) == 0:
        logging.warning("UniProt: Cellular location of UniProtID {} has not been found.".format(uniprot_id))
        return ""
    elif len(df) > 1:
        raise ValueError("UniProt: UniProtID {} returns ambiguous results.".format(uniprot_id))
    else:
        return df["Subcellular location [CC]"].iloc[0]


def encode_sublocation(sublocation):
    # One-hot encoding vector of cellular sublocations
    encoded_sublocation = [0 for _ in range(len(sublocations_dict))]
    for idx, loc in enumerate(sublocations_dict):
        if loc in sublocation:
            encoded_sublocation[idx] = 1
    return encoded_sublocation


def get_sublocation(uniprot_id):
    res = request_sublocation_from_uniprot(uniprot_id)
    return encode_sublocation(res)


def get_sublocation_features_length():
    # Get length of the sublocation one-hot encoded vector
    return len(sublocations_dict)


def GeneID_to_UniProtID(gene_ids, return_list=False):
    # Retrieve UniProtID from GeneID
    result = subprocess.check_output(
        f"curl --form 'from=GeneID' --form 'to=UniProtKB' --form 'ids={gene_ids}' "
        + "https://rest.uniprot.org/idmapping/run",
        shell=True,
    )
    job_id = json.loads(result)["jobId"]

    # Give time to UniProt to execute the job
    check_id_mapping_results_ready(job_id)

    url = "https://rest.uniprot.org/idmapping/uniprotkb/results/{}/?fields=accession&format=tsv&size=500".format(
        job_id
    )
    df = pd.read_csv(url, delimiter="\t").rename(columns={"From": "GeneID", "Entry": "UniProtID"})

    if return_list:
        return df.Entry.to_list()
    else:
        return df


def UniProtID_to_GeneID(uniprot_ids, return_list=False):
    # Retrieve GeneID from UniProtID - CURRENTLY NOT FUNCTIONAL (UniProt Issues)
    result = subprocess.check_output(
        "curl --form 'from=UniProtKB_AC' --form 'to=GeneID' "
        + "--form 'ids={}' https://rest.uniprot.org/idmapping/run".format(",".join(uniprot_ids)),
        shell=True,
    )
    job_id = json.loads(result)["jobId"]

    # Give time to UniProt to execute the job
    check_id_mapping_results_ready(job_id)

    url = "https://rest.uniprot.org/idmapping/uniprotkb/results/{}/?fields=accession&format=tsv&size=500".format(
        job_id
    )
    df = pd.read_csv(url, delimiter="\t").rename(columns={"From": "UniProtID", "Entry": "GeneID"})

    if return_list:
        return df.Entry.to_list()
    else:
        return df


def check_id_mapping_results_ready(job_id):
    # Code snipet to check if UniProt job is completed
    def check_response(response):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(response.json())
            raise

    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")
        check_response(request)
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] == "RUNNING":
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(request["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])


def exclude_nonhuman_proteins(proteins, json_path=None):
    # Check which protein is human, and exclude the ones that are not
    if (
        json_path
        and os.path.isfile(json_path)
        and "human_proteins" in json.loads(open(json_path, encoding="utf-8").read())
    ):
        return json.loads(open(json_path, encoding="utf-8").read())["human_proteins"]
    else:
        human_proteins = []
        for protein in proteins:
            try:
                url = "https://rest.uniprot.org/uniprotkb/search?query=accession:"
                +"{}&format=tsv&fields=organism_id".format(protein)
                df = pd.read_csv(url, delimiter="\t")
                if len(df) == 0:
                    logging.warning(
                        "UniProt: UniProtID {} has been redirected - please manually update your database!".format(
                            protein
                        )
                    )
                elif len(df) > 1:
                    raise ValueError("UniProt: UniProtID {} returns ambiguous results.".format(protein))
                else:
                    if df["Organism (ID)"].iloc[0] == 9606:  # Protein is human
                        human_proteins.append(protein)
            except Exception:
                logging.warning("UniProt: UniProtID {} is invalid!".format(protein))
                continue

        if json_path:
            dump_to_json({"human_proteins": human_proteins}, json_path)
        return human_proteins


def download_uniprot_fasta(proteins, data_folder):
    # Download fasta files of protein sequences
    print("Downloading FASTA files from UniProt...")
    for uniprot_id in proteins:
        if not os.path.isfile(os.path.join(data_folder, "fasta", uniprot_id + ".fasta")):
            url = (
                "https://rest.uniprot.org/uniprotkb/search?query="
                + "{}+AND+organism_id:9606&format=fasta&compressed=false".format(uniprot_id)
            )
            fasta = requests.get(url).text

            fasta_dir = os.path.join(data_folder, "fasta")
            if not os.path.isdir(fasta_dir):
                os.mkdir(fasta_dir)
            with open(os.path.join(fasta_dir, uniprot_id + ".fasta"), "w+") as f:
                f.write(fasta)
            f.close()
    print("Download completed!")


def PDBCode_to_UniProtID(pdbcodes, return_list=False):
    # Retrieve UniProtID from PDB Code
    result = subprocess.check_output(
        "curl --form 'from=PDB' --form 'to=UniProtKB' --form 'ids={}' https://rest.uniprot.org/idmapping/run".format(
            ",".join(pdbcodes)
        ),
        shell=True,
    )
    job_id = json.loads(result)["jobId"]

    # Give time to UniProt to execute the job
    check_id_mapping_results_ready(job_id)

    url = "https://rest.uniprot.org/idmapping/uniprotkb/results/{}/?fields=accession&format=tsv&size=500".format(
        job_id
    )
    df = pd.read_csv(url, delimiter="\t").rename(columns={"From": "PDBCode", "Entry": "UniProtID"})

    if return_list:
        return df.Entry.to_list()
    else:
        return df


def retrieve_protein_classes(proteins):
    # Retrieve Protein classes from UniProtID using UniProt keywords
    # This file has been downloaded directly from the UniProt website
    uniprot_keyword_df = pd.read_csv("uniprot-keywords-table.tsv", delimiter="\t")
    keyword_categories = [
        "Biological process",
        "Molecular function",
        "Disease",
        "Cellular component",
    ]  # We are interested in 4 different categories only
    category2keywords = {
        c: uniprot_keyword_df[uniprot_keyword_df.Category == c].Name.unique().tolist() for c in keyword_categories
    }
    protein_class_df = pd.DataFrame(columns=keyword_categories, index=proteins)
    for protein in proteins:
        url = (
            "https://rest.uniprot.org/uniprotkb/search?query=accession:"
            + "{}+AND+organism_id:9606&format=tsv&fields=keyword".format(protein)
        )
        df = pd.read_csv(url, delimiter="\t")
        keywords_protein_list = df.Keywords.loc[0].split(";")
        protein_class_df.loc[protein] = {
            c: [k for k in keywords_protein_list if k in category2keywords[c]] for c in keyword_categories
        }
    return protein_class_df
