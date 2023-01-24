# Incomplete Text Classification
Authors: Luning Yang, Yacun Wang<br>
Mentor: Jingbo Shang

In this project, we aim to design a text classification model that could suggest class names not belonging to the training corpus to unseen documents, and classify documents into a full set of class names.

### Data
#### Data Information
- The datasets used in the experiments can be found on [Google Drive](https://drive.google.com/drive/folders/1kf3AXpKbwbZuQhcVSiaMzCiaSrWTdO7i?usp=sharing).
- The datasets used in the experiments are: `nyt-fine`, `Reddit`, `DBPedia`

#### Get Data
- [**DSMLP Users**] For the 3 datasets provided, convenient Linux commands to download and get the data are provided in the [documentation of raw data](data/raw/). Please run the commands in the **repository root directory**.
- Generally, under Linux command line, for any Google Drive zip file, 
  - Follow the `wget` [tutorial](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99)
    - Find the Large File section (highlighted code section towards the end)
    - Paste the `<FILEID>` from the `zip` file **sharing link** found on Google Drive
    - Change the `<FILENAME>` to your data title
  - Run `cd <dir>` to change directory into the data directory
  - Run `unzip -o <zip name>` to unzip the data
  - Run `rm <zip name>` to avoid storing too many objects in the container
  - Run `cd <root>` to change directory back to your working directory
  - Run `mkdir <data>` to create the processed data directory
- Under non-command line, go to the Google Drive link, download the zip directly, place the files according to the requirements in the **Data Format** section, and manually created the directory needed for processed files. See the **File Outline** section for example.

#### Data Format
- Raw Data: Each dataset must contain a `df.pkl` placed in `data/raw/`. The file should be a compressed Pandas DataFrame using `pickle` containing two columns: `sentence` (for documents) and `label` (for the corresponding label).
- Processed Data: 
  - The corpus will be processed after the first run, and processed files will be placed in `data/processed`.
  - The processed file will be directly loaded for subsequent runs.
