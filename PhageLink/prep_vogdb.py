import os
from Bio.SeqIO import SeqRecord
from Bio import SeqIO

def merge_vogs(folder_path, output_file):
    '''
    Consolidate all records from .faa files in a folder into a single FASTA file.
    Each record ID will be the file name (without extension).

    Args:
        folder_path: Path to the folder containing .faa files.
        output_file: Path to the output consolidated FASTA file.

    Returns:
        None
    '''
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    with open(output_file, "w") as out_handle:
        for filename in os.listdir(folder_path):
            if filename.endswith(".faa"):
                file_path = os.path.join(folder_path, filename)
                file_id = os.path.splitext(filename)[0]  # Filename without extension
                
                # Read the sequence record from the file
                record = next(SeqIO.parse(file_path, "fasta"))
                record.id = file_id
                record.description = ""  # Clear the description
                
                # Write the record to the output file
                SeqIO.write(record, out_handle, "fasta")
    
    print(f"Consolidated FASTA file created at {output_file}")