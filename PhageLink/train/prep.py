import logging
import os
import subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
from PhageLink.utils import running_message, run_command, read_fasta


@running_message
def prep_vogdb(vogdb_path: str):
    '''
    Prepare the mmseqs database of VOGs.
    '''
    vogdb_dir = os.path.dirname(vogdb_path)
    mmseqs_db_path = os.path.join(vogdb_dir, "vogdb_mmseqs")

    if os.path.exists(mmseqs_db_path) and os.path.getsize(mmseqs_db_path) > 0:
        logging.info(f"VOG database already prepared at: {mmseqs_db_path}")
        return mmseqs_db_path

    logging.info("Preparing the VOG database using mmseqs.")
    os.makedirs(vogdb_dir, exist_ok=True)
    try:
        cmd = f"mmseqs createdb {vogdb_path} {mmseqs_db_path} --dbtype 1"
        run_command(cmd)
        logging.info(f"VOG database prepared at: {mmseqs_db_path}")
    except Exception as e:
        logging.error(f"Failed to prepare the VOG database:")
        raise e
    return mmseqs_db_path


@running_message
def prep_nucleotide_db(fasta_path: str, output_dir: str):
    '''
    Prepare the mmseqs database for nucleotide sequences inside the output directory.
    '''
    mmseqs_db_path = os.path.join(output_dir, "nucleotide_mmseqs")

    if os.path.exists(mmseqs_db_path) and os.path.getsize(mmseqs_db_path) > 0:
        logging.info(f"Nucleotide database already prepared at: {mmseqs_db_path}")
        return mmseqs_db_path

    logging.info("Preparing the nucleotide database using mmseqs.")
    os.makedirs(output_dir, exist_ok=True)
    try:
        cmd = f"mmseqs createdb {fasta_path} {mmseqs_db_path} --dbtype 2"
        run_command(cmd)
        logging.info(f"Nucleotide database prepared at: {mmseqs_db_path}")
    except Exception as e:
        logging.error("Failed to prepare the nucleotide database:")
        raise e

    return mmseqs_db_path


@running_message
def mmseqs_search(query_db, ref_db, outdir, tmp_path, threads, memory):
    '''
    Search the query database against the reference database using mmseqs.
    '''
    if not os.path.exists(f"{outdir}/network.m8"):
        cmd = f"mmseqs search {query_db} {ref_db} {outdir}/network_int {tmp_path} --threads {threads} --split-memory-limit {memory}"
        run_command(cmd, shell=True, check=True)
        cmd2 = f"mmseqs convertalis {query_db} {ref_db} {outdir}/network_int {outdir}/network.m8"
        run_command(cmd2, shell=True, check=True)
    return f"{outdir}/network.m8"


def select_hallmark_genes(presence_absence, top_n=3):
    '''
    Select hallmark genes from the presence-absence table by picking the top N most common queries in each iteration.
    '''
    pa_table = presence_absence.copy()
    hallmark_genes = []

    while not pa_table.empty:
        query_sums = pa_table.sum(axis=0)

        top_queries = query_sums.nlargest(top_n).index.tolist()
        hallmark_genes.extend(top_queries)

        subjects_with_queries = pa_table.index[pa_table[top_queries].any(axis=1)]

        pa_table.drop(index=subjects_with_queries, inplace=True)

        pa_table.drop(columns=top_queries, inplace=True)

    return hallmark_genes


def extract_and_align_hallmark_genes(
    hallmark_genes: list,
    search_df: pd.DataFrame,
    seqs_path: str,
    output_dir: str,
    threads: int
):
    '''
    For each hallmark gene, extract the sequences, align them using MAFFT via command line, and generate a distance matrix.
    '''
    logging.info("Extracting and aligning hallmark genes")

    # Create directories
    fasta_dir = os.path.join(output_dir, "hallmark_fastas")
    align_dir = os.path.join(output_dir, "hallmark_alignments")
    distance_dir = os.path.join(output_dir, "hallmark_distances")

    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(distance_dir, exist_ok=True)

    # Read the nucleotide sequences
    sequences = {record.id: record for record in read_fasta(seqs_path)}

    for gene in hallmark_genes:
        distance_file = os.path.join(distance_dir, f"{gene}_distance.tsv")
        if os.path.exists(distance_file):
            logging.info(f"Distance matrix for gene {gene} already exists. Skipping computation.")
            continue  # Skip to the next gene

        logging.info(f"Processing hallmark gene: {gene}")

        # Get the hits for this gene
        gene_hits = search_df[search_df['query'] == gene]

        # Extract sequences for subjects (targets)
        records = []
        for idx, row in gene_hits.iterrows():
            subject_id = row['target']
            qstart = int(row['qstart']) - 1
            qend = int(row['qend'])
            seq_record = sequences.get(subject_id)
            if seq_record:
                # Extract the sequence region
                seq_fragment = seq_record.seq[qstart:qend]
                new_record = SeqRecord(
                    seq=seq_fragment,
                    id=subject_id,
                    description=''
                )
                records.append(new_record)
            else:
                logging.warning(f"Sequence for subject {subject_id} not found in input sequences.")

        if not records:
            logging.warning(f"No sequences found for hallmark gene {gene}. Skipping.")
            continue

        # Save the extracted sequences to a FASTA file
        fasta_file = os.path.join(fasta_dir, f"{gene}.fasta")
        with open(fasta_file, 'w') as f_out:
            SeqIO.write(records, f_out, 'fasta')

        # Align the sequences using MAFFT
        alignment_file = os.path.join(align_dir, f"{gene}_aligned.fasta")
        cmd = f"mafft --quiet --auto --thread {threads} {fasta_file}"

        try:
            # Run MAFFT and capture the output
            result = run_command(cmd)
            # Write the alignment
            with open(alignment_file, 'w') as f_out:
                f_out.write(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"MAFFT failed for gene {gene}: {e}")
            continue

        # Generate distance matrix
        alignment = AlignIO.read(alignment_file, 'fasta')
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(alignment)

        # Save the distance matrix
        with open(distance_file, 'w') as f_out:
            f_out.write('\t' + '\t'.join(dm.names) + '\n')
            for name1 in dm.names:
                distances = [f"{dm[name1, name2]:.5f}" for name2 in dm.names]
                f_out.write(name1 + '\t' + '\t'.join(distances) + '\n')

        logging.info(f"Processed hallmark gene: {gene}")


def combine_distance_matrices(
    distance_dir: str,
    output_dir: str
):
    '''
    Combine all individual distance matrices into a single comprehensive weighted adjacency matrix.
    '''
    import os
    import pandas as pd
    import numpy as np
    from functools import reduce

    combined_distance_file = os.path.join(output_dir, 'combined_distance_matrix.tsv')
    weighted_adjacency_matrix_file = os.path.join(output_dir, 'weighted_adjacency_matrix.tsv')

    if os.path.exists(weighted_adjacency_matrix_file):
        logging.info("Combined weighted adjacency matrix already exists. Skipping recomputation.")
        return

    logging.info("Combining individual distance matrices into a comprehensive weighted adjacency matrix")

    # List all distance matrix files
    distance_files = [os.path.join(distance_dir, f) for f in os.listdir(distance_dir) if f.endswith('_distance.tsv')]

    if not distance_files:
        logging.warning("No distance matrices found to combine.")
        return

    # Initialize a list to hold DataFrames
    distance_dfs = []

    for file in distance_files:
        # Read the distance matrix into a DataFrame
        df = pd.read_csv(file, sep='\t', index_col=0)
        distance_dfs.append(df)

    # Align all DataFrames on rows and columns
    sum_distances = reduce(lambda x, y: x.add(y, fill_value=0), distance_dfs)
    counts = reduce(lambda x, y: x.add(~y.isna(), fill_value=0), distance_dfs)

    # avoid division by zero
    counts.replace(0, np.nan, inplace=True)

    # Compute average distances
    average_distance = sum_distances.divide(counts)

    # Replace NaN values
    average_distance = average_distance.fillna(1.0)

    average_distance.to_csv(combined_distance_file, sep='\t')
    logging.info(f"Combined distance matrix saved to {combined_distance_file}")

    weights = 1 - average_distance
    weights = weights.clip(lower=0.0, upper=1.0)

    # no self loops
    np.fill_diagonal(weights.values, 0)

    weights.to_csv(weighted_adjacency_matrix_file, sep='\t')
    logging.info(f"Combined weighted adjacency matrix saved to {weighted_adjacency_matrix_file}")


def calculate_hypergeometric_adjacency(
    presence_absence: pd.DataFrame,
    output_dir: str,
    p_value_threshold: float = 0.05
):
    '''
    Calculate a binary adjacency matrix from the presence-absence matrix using the hypergeometric distribution.
    '''
    import os
    import pandas as pd
    import numpy as np
    import torch

    adjacency_matrix_file = os.path.join(output_dir, 'hypergeometric_adjacency_matrix.tsv')

    # Check if the adjacency matrix already exists
    if os.path.exists(adjacency_matrix_file):
        logging.info("Hypergeometric adjacency matrix already exists. Skipping recomputation.")
        return

    logging.info("Calculating hypergeometric adjacency matrix from presence-absence matrix using PyTorch")

    # Set up device for PyTorch (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    M = presence_absence.shape[1]
    subject_gene_counts = presence_absence.sum(axis=1)
    subjects = presence_absence.index.tolist()

    pa_tensor = torch.tensor(presence_absence.values, dtype=torch.float32, device=device)
    gene_counts = torch.tensor(subject_gene_counts.values, dtype=torch.float32, device=device)
    shared_gene_counts = torch.mm(pa_tensor, pa_tensor.t())

    # Prepare parameters for hypergeometric distribution
    M_tensor = torch.tensor(M, dtype=torch.float32, device=device)
    n = gene_counts.unsqueeze(1)
    N = gene_counts.unsqueeze(0)
    k = shared_gene_counts

    def log_binom(a, b):
        return torch.lgamma(a + 1) - torch.lgamma(b + 1) - torch.lgamma(a - b + 1)

    mean = N * n / M_tensor
    std = torch.sqrt(N * n * (M_tensor - n) * (M_tensor - N) / (M_tensor**2 * (M_tensor - 1) + 1e-10))

    # Compute z-scores
    z = (k - mean) / (std + 1e-10)

    # Compute p-values using the normal distribution's survival function
    from torch.distributions import Normal
    normal_dist = Normal(0, 1)
    p_vals = 1 - normal_dist.cdf(z)

    # Set diagonal to zero
    p_vals.fill_diagonal_(0)
    adjacency_matrix = (p_vals <= p_value_threshold).int()

    # Move tensor back to CPU and convert to numpy
    adjacency_matrix_cpu = adjacency_matrix.cpu().numpy()

    adjacency_df = pd.DataFrame(adjacency_matrix_cpu, index=subjects, columns=subjects)
    adjacency_df.to_csv(adjacency_matrix_file, sep='\t')
    logging.info(f"Hypergeometric adjacency matrix saved to {adjacency_matrix_file}")


def Prepare_data(
    seqs: str,
    map_bitscore_threshold: int,
    reference_data: str,
    output: str,
    threads: int,
    memory: str,
    p_value_threshold: float = 0.05
):
    '''
    Prepare the data for training and generate necessary matrices.
    '''
    logging.info("Preparing the data")
    vogdb_faa_path = f"{reference_data}/vogdb_merged.faa"

    # Prepare mmseqs databases
    vog_mmseqs_db = prep_vogdb(vogdb_faa_path)
    seq_mmseqs_db = prep_nucleotide_db(seqs, output)

    # Run mmseqs search
    search_output_dir = f"{output}/search_output"
    os.makedirs(search_output_dir, exist_ok=True)

    tmp = f"{output}/tmp"
    os.makedirs(tmp, exist_ok=True)

    search_output = mmseqs_search(vog_mmseqs_db, seq_mmseqs_db, search_output_dir, tmp, threads, memory)
    columns = ["query", "target", "pident", "alnlen", "mismatch", "numgapopen",
               "qstart", "qend", "tstart", "tend", "evalue", "bitscore"]
    search_df = pd.read_csv(search_output, sep='\t', header=None, names=columns)

    pres_abs_path = f"{output}/presence_absence.tsv"

    filtered_df = search_df[search_df["bitscore"] >= map_bitscore_threshold]
    filtered_df = filtered_df.sort_values(by=["query", "target", "bitscore"], ascending=[True, True, False])
    filtered_df = filtered_df.drop_duplicates(subset=["query", "target"], keep="first")

    # Generate the presence-absence table
    presence_absence = pd.crosstab(filtered_df["target"], filtered_df["query"])
    presence_absence.to_csv(pres_abs_path, sep='\t')
    logging.info(f"Presence-absence table saved to {pres_abs_path}")

    hallmark_genes_file = f"{output}/hallmark_genes.txt"
    if not os.path.exists(hallmark_genes_file):
        hallmark_genes = select_hallmark_genes(presence_absence)
        with open(hallmark_genes_file, 'w') as f:
            for gene in hallmark_genes:
                f.write(f"{gene}\n")
        logging.info(f"Hallmark genes selected and saved to {hallmark_genes_file}")
    else:
        with open(hallmark_genes_file, 'r') as f:
            hallmark_genes = [line.strip() for line in f]
        logging.info(f"Hallmark genes loaded from {hallmark_genes_file}")

    #Extract sequences for each hallmark gene
    extract_and_align_hallmark_genes(
        hallmark_genes,
        filtered_df,
        seqs,
        output,
        threads
    )

    # Combine all distance matrices into a weighted adjacency matrix
    distance_dir = os.path.join(output, "hallmark_distances")
    combine_distance_matrices(
        distance_dir=distance_dir,
        output_dir=output
    )

    #Calculate binary hypergeometric adjacency matrix
    calculate_hypergeometric_adjacency(
        presence_absence=presence_absence,
        output_dir=output,
        p_value_threshold=p_value_threshold
    )
