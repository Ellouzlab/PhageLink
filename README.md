
# PhageLink

Classify phage by predicting taxonomic links

### Installation
1. Download mamba
2. Clone repository
```
git clone https://github.com/Ellouzlab/PhageLink
```
3. Change Directory into PhageLink

```
cd PhageLink
```

4. Create environment and download dependencies
```
mamba env create -n PhageLink --file env.yml
```

5. Install PhageLink
```
python -m pip install -e .
```

### Download Databases

Located on the following google drive: https://drive.google.com/drive/folders/1RnZT6LDQS2zvAFQ5iMExymeuq48C41Cz?usp=sharing

Download and unzip. Navigate into the directory and unzip reference.

### Run tests

The following will run tests

```
PhageLink train \
--csv path_to_csv_in_downloaded_data \
--seqs path_to_sequence_fasta_in_downloaded_data \
--reference_data path_to_unzipped_reference_data

```

Currently only the train module is setup. If you need more instructions type the following:
```
PhageLink train -h
```

### Considerations

Currently this program only works on linux with cuda. I will fix this later.
