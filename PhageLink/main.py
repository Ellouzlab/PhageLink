import argparse
import os
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
    train_parser.add_argument("--map_bitscore_threshold", help="mmseqs minimum bitscore for mapping", default=50)
    train_parser.add_argument("--reference_data", help="reference data directory", default="Data/reference")
    train_parser.add_argument("--output", help="output directory", default="Data/output")
    train_parser.add_argument("--threads", help=f"number of threads to use. Default ({cpu_count()-1})", default=cpu_count()-1)
    train_parser.add_argument("--memory", help="memory to use", default="50G")
    train_parser.add_argument("--draw", help="export the network for Cytoscape", action="store_true")

    return args.parse_args()


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

        logging.info("Reading hypergeometric adjacency matrix")
        hyper_adj_df = pd.read_csv(adjacency_matrix_file, sep="\t", index_col=0)
        logging.info("Reading weighted adjacency matrix")
        w_comb_adj_df = pd.read_csv(weighted_adjacency_matrix_file, sep="\t", index_col=0)
        logging.info("Done reading matrices")

        if arguments.draw:
            from PhageLink.train.draw import draw_network
            draw_network(hyper_adj_df, w_comb_adj_df, arguments.output, arguments.csv)
        
        output_taxa_path = os.path.join(arguments.output, 'taxonomy_df.csv')
        if not os.path.exists(output_taxa_path): 
            from PhageLink.train.get_answers import create_taxonomy_similarity_matrix
            taxa_df = create_taxonomy_similarity_matrix(arguments.csv, output_taxa_path)
        else:
            taxa_df = pd.read_csv(output_taxa_path, index_col=0)
        
        from PhageLink.train.train_gat import train_gat_edge_model
        from PhageLink.train.train_sage import train_graphsage_edge_model
        model = train_graphsage_edge_model(hyper_adj_df, w_comb_adj_df, taxa_df)
        model = train_gat_edge_model(hyper_adj_df, w_comb_adj_df, taxa_df)

