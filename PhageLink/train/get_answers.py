import pandas as pd
import numpy as np

def create_taxonomy_similarity_matrix(csv_file, output_csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    accessions = df['Accession'].values
    N = len(accessions)
    
    # Initialize the similarity matrix with zeros
    similarity_matrix = np.zeros((N, N), dtype=int)
    
    # Extract taxonomy columns and fill missing values with unique placeholders
    families = df['Family'].fillna('MISSING_FAMILY').values
    genera = df['Genus'].fillna('MISSING_GENUS').values
    species = df['Species'].fillna('MISSING_SPECIES').values
    
    # Create boolean arrays indicating known values
    family_known = df['Family'].notnull().values
    genus_known = df['Genus'].notnull().values
    species_known = df['Species'].notnull().values
    
    # Compare families
    family_match = np.equal.outer(families, families)
    both_family_known = np.logical_and.outer(family_known, family_known)
    family_differ = np.logical_and(both_family_known, ~family_match)
    family_same = np.logical_and(both_family_known, family_match)
    
    # Assign similarity scores based on family comparison
    similarity_matrix[family_differ] = 1
    similarity_matrix[family_same] = 2
    
    # Compare genera and assign similarity score of 3 where genera match
    genus_match = np.equal.outer(genera, genera)
    both_genus_known = np.logical_and.outer(genus_known, genus_known)
    genus_same = np.logical_and(both_genus_known, genus_match)
    similarity_matrix[genus_same] = 3
    
    # Compare species and assign similarity score of 4 where species match
    species_match = np.equal.outer(species, species)
    both_species_known = np.logical_and.outer(species_known, species_known)
    species_same = np.logical_and(both_species_known, species_match)
    similarity_matrix[species_same] = 4
    
    # Create a DataFrame for the similarity matrix with accessions as labels
    similarity_df = pd.DataFrame(similarity_matrix, index=accessions, columns=accessions)
    
    # Save the similarity matrix to a CSV file
    similarity_df.to_csv(output_csv_path)
