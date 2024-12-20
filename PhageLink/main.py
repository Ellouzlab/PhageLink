# main.py

import argparse
import os
import logging
import pandas as pd
from multiprocessing import cpu_count
from PhageLink.verify import verify
from PhageLink.utils import init_logging
from PhageLink.check_repeats import check_repeats


def argparser():
    args = argparse.ArgumentParser(description="Classify viruses")
    args.add_argument('-v', '--version', action='version', version="0.0.0")
    subparsers = args.add_subparsers(dest='command', help='sub-command help')

    train_parser = subparsers.add_parser('train', help='Train GAT')
    train_parser.add_argument("--seqs", help="sequence fasta", required=True)
    train_parser.add_argument("--csv", help="Taxonomy CSV", required=True)
    train_parser.add_argument("--map_bitscore_threshold", type=int, help="mmseqs minimum bitscore for mapping", default=50)
    train_parser.add_argument("--reference_data", help="reference data directory", default="Data/reference")
    train_parser.add_argument("--output", help="output directory", default="Data/output")
    train_parser.add_argument("--threads", type=int, help=f"number of threads to use. Default ({cpu_count()-1})", default=cpu_count()-1)
    train_parser.add_argument("--memory", help="memory to use", default="50G")
    train_parser.add_argument("--draw", help="export the network for Cytoscape", action="store_true")

    return args.parse_args()

def align_data(taxa_df, presence_absence_df, hyper_adj_df, weighted_adj_df, family_labels, genus_labels):
    """
    Aligns all input data frames and series to ensure consistency across indices.
    Now includes genus_labels for alignment.
    """
    logging.info("Aligning data across all matrices and labels.")
    shared_indices = list(
        set(taxa_df.index)
        & set(presence_absence_df.index)
        & set(hyper_adj_df.index)
        & set(weighted_adj_df.index)
        & set(family_labels.index)
        & set(genus_labels.index)
    )

    logging.info(f"Found {len(shared_indices)} shared indices across all datasets.")

    taxa_df = taxa_df.loc[shared_indices, shared_indices]
    presence_absence_df = presence_absence_df.loc[shared_indices]
    hyper_adj_df = hyper_adj_df.loc[shared_indices, shared_indices]
    weighted_adj_df = weighted_adj_df.loc[shared_indices, shared_indices]
    family_labels = family_labels.loc[shared_indices]
    genus_labels = genus_labels.loc[shared_indices]

    return taxa_df, presence_absence_df, hyper_adj_df, weighted_adj_df, family_labels, genus_labels


def main():
    arguments = argparser()
    verify(arguments)

    logging_folder = f"{arguments.output}/logs"
    os.makedirs(logging_folder, exist_ok=True)
    os.makedirs(arguments.output, exist_ok=True)

    vogdb_dir_path = f"{arguments.reference_data}/vog.faa/faa"
    vogdb_merged_path = f"{arguments.reference_data}/vogdb_merged.faa"

    if not os.path.exists(vogdb_merged_path) or os.path.getsize(vogdb_merged_path) == 0:
        print("Merging VOGs")
        from PhageLink.prep_vogdb import merge_vogs
        merge_vogs(vogdb_dir_path, vogdb_merged_path)

    check_repeats(arguments)

    if arguments.command == "train":
        from PhageLink.train.prep import Prepare_data

        init_logging(f"{logging_folder}/train.log")
        adjacency_matrix_file = os.path.join(arguments.output, 'hypergeometric_adjacency_matrix.tsv')
        weighted_adjacency_matrix_file = os.path.join(arguments.output, 'weighted_adjacency_matrix.tsv')
        presence_absence_file = os.path.join(arguments.output, 'presence_absence.tsv')  # Added presence_absence path

        if not os.path.exists(adjacency_matrix_file) or not os.path.exists(weighted_adjacency_matrix_file):
            Prepare_data(
                arguments.seqs,
                arguments.map_bitscore_threshold,
                arguments.reference_data,
                arguments.output,
                arguments.threads,
                arguments.memory
            )

        import pandas as pd
        import logging
        import torch

        logging.info("Reading hypergeometric adjacency matrix")
        hyper_adj_df = pd.read_csv(adjacency_matrix_file, sep="\t", index_col=0)
        logging.info("Reading weighted adjacency matrix")
        w_comb_adj_df = pd.read_csv(weighted_adjacency_matrix_file, sep="\t", index_col=0)
        logging.info("Done reading matrices")

        if arguments.draw:
            from PhageLink.train.draw import draw_network
            draw_network(hyper_adj_df, w_comb_adj_df, arguments.output, arguments.csv)
        
        output_taxa_path = os.path.join(arguments.output, 'taxonomy_similarity_matrix.csv')
        if not os.path.exists(output_taxa_path): 
            from PhageLink.train.get_answers import create_taxonomy_similarity_matrix
            create_taxonomy_similarity_matrix(arguments.csv, output_taxa_path)
            taxa_df = pd.read_csv(output_taxa_path, index_col=0)
        else:
            taxa_df = pd.read_csv(output_taxa_path, index_col=0)
        
        # Load presence-absence matrix
        presence_absence_path = os.path.join(arguments.output, 'presence_absence.tsv')
        if not os.path.exists(presence_absence_path):
            logging.error(f"Presence-absence file not found at {presence_absence_path}.")
            raise FileNotFoundError(f"Presence-absence file not found at {presence_absence_path}.")
        presence_absence_df = pd.read_csv(presence_absence_path, sep='\t', index_col=0)
        logging.info("Loaded gene presence-absence matrix.")
        
        # Define 'genomes' as the intersection of indices across all DataFrames
        genomes = list(set(hyper_adj_df.index) & set(w_comb_adj_df.index) & set(taxa_df.index) & set(presence_absence_df.index))
        genomes.sort()
        logging.info(f"Number of genomes after intersection: {len(genomes)}")
        
        # Subset presence_absence_df to genomes
        presence_absence_df = presence_absence_df.loc[genomes]
        logging.info(f"Presence-absence matrix subset to {len(genomes)} genomes.")
        
        # Load original taxonomy data to extract 'Family' and 'Genus' labels
        logging.info("Loading original taxonomy data to extract 'Family' and 'Genus' labels.")
        original_taxonomy_df = pd.read_csv(arguments.csv)
        
        # Ensure 'Accession' is the index and align with 'genomes'
        original_taxonomy_df = original_taxonomy_df.set_index('Accession').loc[genomes]
        
        # Extract 'Family' and 'Genus' labels
        family_labels = original_taxonomy_df['Family']
        genus_labels = original_taxonomy_df['Genus']  # Assuming the CSV has a column named 'Genus'
        logging.info("Extracted 'Family' and 'Genus' labels.")
        
        from PhageLink.train.trainer import train_model
        # Align data including genus_labels
        taxa_df, presence_absence_df, hyper_adj_df, w_comb_adj_df, family_labels, genus_labels = align_data(
            taxa_df, presence_absence_df, hyper_adj_df, w_comb_adj_df, family_labels, genus_labels
        )

        # Now pass genus_labels to train_model as well
        train_model(
            hyper_adj_df=hyper_adj_df,
            weighted_adj_df=w_comb_adj_df,
            taxa_df=taxa_df,
            presence_absence_df=presence_absence_df,
            family_labels=family_labels,
            genus_labels=genus_labels,   # Passing genus_labels here
            output_dir=arguments.output
        )


if __name__ == "__main__":
    main()
