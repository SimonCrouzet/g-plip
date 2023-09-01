# G-PLIP: Knowledge graph neural network for structure-free protein-ligand affinity prediction

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![RDKit badge](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
[![License CC BY-NC-SA 4.0](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
[![DOI:10.1101/2023.09.01.555977](https://zenodo.org/badge/DOI/10.1101/2023.09.01.555977.svg)](https://doi.org/10.1101/2023.09.01.555977)
[![Citations](https://api.juleskreuer.eu/citation-badge.php?doi=10.1101/2023.09.01.555977)](https://juleskreuer.eu/projekte/citation-badge/)

![](gplip_pipeline.pdf)

This repository contains a reference implementation to process the data, build the graph, to train and apply the graph neural network model introduced in the paper "G-PLIP: Knowledge graph neural network for structure-free protein-ligand affinity prediction" by Simon J. Crouzet, Anja Maria Lieberherr, Kenneth Atz, Tobias Nilsson, Lisa Sach-Peltason, Alex T. Müller, Matteo Dal Peraro, and Jitao David Zhang.


## 1. Environment
Create and activate the gplip environment. 

```
conda env create -f requirements.yml
conda activate g-plip
```

Due to its size, the model is running on CPU. Thus, you don't need to install cuda-related packages.

With its minimal architecture, the model is still able to train on CPUs in a relatively short time. You'll however need an important amount of available RAM.

## 2. Data

The `data/` directory needs:
 - The PDBBind v2020 data, which can be retrieved from [the PDBBind website](http://pdbbind.org.cn/download.php). You need both the "general set minus refined set (2)" and the "refined set (3)", to unzip them and merge the two in the folder `data/v2020-complete-PL`
 - The PDBBind CASF 2016 Benchmark set, which can be retrieved from [the PDBBind website](http://pdbbind.org.cn/casf.php). You need the "CASF-2016" data, unzipped in the folder `data/CASF-2016`
 - The `Mapping_HumanProteins_20221006.tsv`, provided, giving a mapping between GeneID, UniProtID and EnsemblID for human proteins
 - A PPI database giving pairs of GeneIDs interacting. By default, we used the database from the publication "Large-scale analysis of disease pathways in the human interactome" by M. Agrawal, M. Zitnik & J. Leskovec, [available here](https://snap.stanford.edu/biodata/datasets/10000/10000-PP-Pathways.html)
 - Gene expression data, retrieved by default [from the Human Protein Atlas](https://www.proteinatlas.org/about/download), where we took the "RNA consensus tissue gene data (4)"

We provide the prebuild graph from the PDBBind data using the default settings in `configs/config.ini`:
 - [Graph for the 2016 CASF Core Set](https://drive.google.com/file/d/1iae-QR7hU7tb_CDQ9UMZrx2-ablgI2TV/view?usp%253Dsharing)
 - [Graph for the 2019 Temporal Hold-out Set](https://drive.google.com/file/d/1TY9ay1i-31VXO7SW5B9Wi35x_g1WoEK5/view?usp%253Dsharing)

(please keep the name intact)

N.B.: Note you still need the raw data - G-PLIP is performing a file check first.

## 3. Graph building
With the raw data correctly imported, you can proceed to the graph building with `python -m scripts.construct_pdbbind.py`.


## 4. Model training
The model can be trained on the PDBBind dataset, using `python -m scripts.pdbbind_pipeline.py [args]`.
 - By default, evaluation is done on the 2016 CASF Core Set. For an evaluation on the 2019 Temporal Hold-out Set, use `--split_type temporal`
 - To use the `refined` or `other` subset of PDBBind rather than the `complete`, use `--pdbbind_set refined` or `--pdbbind_set other`. For that, you will need the corresponding data in the folders `data/v2020-refined-PL` and `data/v2020-other-PL`, as present by default in the PDBBind zip files.


## 10. License
The software was developed at F. Hoffmann - La Roche Ltd. and is licensed by the license CreativeCommons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0), i.e. described in `LICENSE`.


## 11. Citation

```
@misc{crouzet_pliprediction_2023,
	title = {G-{PLIP}: {Knowledge} graph neural network for structure-free protein-ligand affinity prediction},
	doi = {10.1101/2023.09.01.555977},
	publisher = {bioRxiv},
	author = {Crouzet, Simon J. and Lieberherr, Anja Maria and Atz, Kenneth and Nilsson, Tobias and Sach-Peltason, Lisa and M{\"u}ller, Alex T. and Peraro, Matteo Dal and Zhang, Jitao David},
	month = sep,
	year = {2023},
}
```
