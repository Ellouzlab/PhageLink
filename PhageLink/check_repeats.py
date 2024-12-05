import sys
from PhageLink.utils import read_fasta



def check_repeated_sequence_ids(fasta_file):
    '''
    Check if there are repeated sequence IDs in the provided FASTA file.

    Args:
        fasta_file: Path to the FASTA file.

    Returns:
        A list of duplicate sequence IDs if any, otherwise an empty list.
    '''
    print(f"Checking for repeated sequence IDs in: {fasta_file}")
    
    # Read the FASTA file using the read_fasta function
    records = read_fasta(fasta_file)
    
    seq_ids = set()
    duplicates = set()

    for record in records:
        if record.id in seq_ids:
            duplicates.add(record.id)
        else:
            seq_ids.add(record.id)

    if duplicates:
        print(f"Found repeated sequence IDs: {duplicates}")
    else:
        print("No repeated sequence IDs found.")

    return list(duplicates)

def check_repeats(arguments):
    '''
    Check for repeated sequence IDs in the provided sequences and VOG database.
    
    Args:
        arguments: The parsed command-line arguments.
    '''
    duplicates = check_repeated_sequence_ids(arguments.seqs)
    if duplicates:
        print(f"Error: Duplicate sequence IDs found in {arguments.seqs}: {duplicates}")
        sys.exit(1)
    
    vogdb_path = f"{arguments.reference_data}/vogdb_merged.faa"
    duplicates = check_repeated_sequence_ids(vogdb_path)
    if duplicates:
        print(f"Error: Duplicate sequence IDs found in {vogdb_path}: {duplicates}")
        sys.exit(1)