import os, sys


def verify(arguments):
    '''
    Verify the arguments passed to the program.
    '''
    subcommands = ["train"]
    if arguments.command not in subcommands:
        print("Please specify a valid command.")
        print("Available commands:", subcommands)
        sys.exit(1)
    
    # Check files
    files_to_check = [arguments.seqs, arguments.csv]
    for file in files_to_check:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            sys.exit(1)
        if os.path.isfile(file) and os.path.getsize(file) == 0:
            print(f"File is empty: {file}")
            sys.exit(1)
    
    # Check directories
    vogdb_path = f"{arguments.reference_data}/vog.faa"
    dirs_to_check = [arguments.reference_data, vogdb_path]
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            sys.exit(1)
        if os.path.isdir(directory) and not any(os.scandir(directory)):
            print(f"Directory is empty: {directory}")
            sys.exit(1)
    
