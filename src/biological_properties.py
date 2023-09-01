import sys, os, itertools, logging
# its win32, maybe there is win64 too?
is_windows = sys.platform.startswith('win')
if is_windows:
	print('N.B.: Windows detected, ESM2 cannot be used - please do not use ESM encodings', file=sys.stderr)
else:
	import esm
import torch
from tqdm import tqdm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from src.external import get_sublocation, get_sublocation_features_length

protein_alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Count k-mers from a protein sequence
# We are not working on reads so perfomances should not be such an issue
def count_khmer(protein_name, data_folder, k, to_list=False):
	fasta = os.path.join(data_folder, 'fasta', '{}.fasta'.format(protein_name))
	try:
		valid = False
		khmer_dict = {}
		for seq_record in SeqIO.parse(open(fasta, "r+"), 'fasta'):
			if protein_name not in seq_record.id:
				logging.warning('Ignoring FASTA sequence: Protein name {} not found in fasta file {}'.format(protein_name, fasta))
				continue
			else:
				if valid:
					logging.error('Ignoring FASTA sequence: Multiple sequences found for protein name {} in fasta file {}'.format(protein_name, fasta))
				valid = True
				seq = str(seq_record.seq)
				khmer_dict = {''.join(comb):0 for comb in itertools.product(protein_alphabet, repeat=k)}
				for i in range(len(seq)-k):
					khmer = seq[i:i+k]
					if khmer not in khmer_dict:
							logging.error('Protein features: khmer {} found in protein {}'.format(khmer, os.path.split(fasta)[-1]))
					else:
						khmer_dict[khmer] += 1
		if not valid:
			raise ValueError('count-khmer: error while reading fasta file {}'.format(fasta))
	except:
		raise ValueError('count-khmer: error while reading fasta file {}'.format(fasta))
	if to_list:
		return list(khmer_dict.values())
	else:
		return khmer_dict

def length_khmer(k):
	# Get length of k-mer vector
	return len({''.join(comb):0 for comb in itertools.product(protein_alphabet, repeat=k)})

def get_protein_gene_expr(target_ensembl, gene_expr_df, tissue='lung'):
	# Retrieve gene expression from an EnsemblGeneID
	if tissue is not None and tissue != 'all':
		gene_expressions = gene_expr_df[(gene_expr_df.Tissue.str.lower().str.contains(tissue.lower())) & (gene_expr_df.Gene.isin(target_ensembl))]
	else:
		gene_expressions = gene_expr_df[gene_expr_df.Gene.isin(target_ensembl)]
	
	
	if len(gene_expressions) == 0:
		return [-1.0] # Flag indicating lack of gene expression value
	else:
		return [gene_expressions.nTPM.median()]

def get_protein_gene_expr_length():
	# Explicitely declare it - might change in the future
	return 1

def get_protein_features(uniprot_id, ensembl_id, gene_expr_df, esm_encodings, data_folder, khmer=True, k_khmer=3, sublocation=True, gene_expr=True, gene_expr_tissue='lung', esm_encoding=True):
	base = []

	if not sublocation and not gene_expr and not khmer:
		# No protein features, return 'Dummy' features
		return [1.]

	if sublocation:
		base = base + get_sublocation(uniprot_id)
	if gene_expr:
		base = base + get_protein_gene_expr(ensembl_id, gene_expr_df, tissue=gene_expr_tissue)
	if khmer:
		base = base + count_khmer(uniprot_id, data_folder, k_khmer, to_list=True)
	if esm_encoding:
		base = base + esm_encodings[uniprot_id]
	return base    

# Check number of protein features
def get_nb_protein_features(khmer=True, k_khmer=3, sublocation=True, gene_expr=True, esm_encoding=True):
	base = 0

	if not sublocation and not gene_expr and not khmer and not esm_encoding:
		return 1
	
	if sublocation:
		base = base + get_sublocation_features_length()
	if gene_expr:
		base = base + get_protein_gene_expr_length()
	if khmer:
		base = base + length_khmer(k_khmer)
	if esm_encoding:
		_, _, _, embedding_dim = get_esm_model()
		base = base + embedding_dim
	return base

def get_union_proteins(uniprotids1, uniprotids2):
	# Get the union of two lists of UniProt IDs
	union_uniprotids = []
	idx_list1 = []
	for idx, uniprot_id in enumerate(uniprotids1):
		if uniprot_id not in union_uniprotids:
			union_uniprotids.append(uniprot_id)
			idx_list1.append(idx)

	idx_list2 = []
	for idx, uniprot_id in enumerate(uniprotids2):
		if uniprot_id not in union_uniprotids:
			union_uniprotids.append(uniprot_id)
			idx_list2.append(idx)
		
	return union_uniprotids, (idx_list1, idx_list2)

def get_esm_model():
	esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
	esm_batch_converter = esm_alphabet.get_batch_converter()
	esm_model.eval()

	embedding_dim = 320 # Not pretty, needs to be changed manually

	return esm_model, esm_alphabet, esm_batch_converter, embedding_dim


def generate_esm_encodings(protein_name_list, to_list=True):
	return {protein_name: get_esm_encoding(protein_name, to_list) for protein_name in tqdm(protein_name_list, desc='Generating ESM2 encodings...')}

def get_esm_encoding(protein_name, to_list = True):
	# Get ESM encoding for a protein
	# ESM is a pretrained model from Facebook Research
	# NB: Encodings are queried one by one, due to memory issues!
	esm_data = []

	fasta = 'data/fasta/{}.fasta'.format(protein_name)
	valid = False
	for seq_record in SeqIO.parse(open(fasta, "r+"), 'fasta'):
		if protein_name not in seq_record.id:
			logging.warning('Ignoring FASTA sequence: Protein name {} not found in fasta file {}'.format(protein_name, fasta))
			continue
		else:
			if valid:
				logging.error('Ignoring FASTA sequence: Multiple sequences found for protein name {} in fasta file {}'.format(protein_name, fasta))
			valid = True
			esm_data.append((protein_name, str(seq_record.seq)))
	if not valid:
		logging.error('ESM encoding: error while reading fasta file {}'.format(fasta))

	esm_model, esm_alphabet, esm_batch_converter, _ = get_esm_model()

	esm_batch_labels, esm_batch_strs, esm_batch_tokens = esm_batch_converter(esm_data)
	esm_batch_lens = (esm_batch_tokens != esm_alphabet.padding_idx).sum(1)

	# Extract per-residue representations (on CPU)
	with torch.no_grad():
		esm_results = esm_model(esm_batch_tokens, repr_layers=[6], return_contacts=True)
	esm_token_representations = esm_results["representations"][6]

	# Generate per-sequence representations via averaging
	# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
	esm_sequence_representations = {}
	for i, tokens_len in enumerate(esm_batch_lens):
		esm_sequence_representations[esm_data[i][0]] = esm_token_representations[i, 1 : tokens_len - 1].mean(0)

	del esm_model # Free up some memory

	assert len(esm_sequence_representations) == 1 # Only one protein per query
	if to_list:
		return esm_sequence_representations[esm_data[0][0]].tolist()
	else:
		return esm_sequence_representations[esm_data[0][0]]

