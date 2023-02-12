# Incomplete Supervision: Text Classification based on a Subset of Labels
Authors: Luning Yang, Yacun Wang<br>
Mentor: Jingbo Shang

In this project, we aim to design a text classification model that could suggest class names not belonging to the training corpus to unseen documents, and classify documents into a full set of class names.

### Environment

- [**DSMLP Users**]: Since the data for this project is large, please run DSMLP launch script using a larger RAM. The suggested command is `launch.sh -i yaw006/incomplete-tc:checkpoint -m 16`. Please **DO NOT** use the default, otherwise Python processes might be killed halfway.
- Other options:
  - Option 1: Run the docker container: `docker run yaw006/incomplete-tc:checkpoint`;
  - Option 2: Install all required packages in `requirements.txt`.

### Data
#### Data Information
- The datasets used in the experiments can be found on [Google Drive](https://drive.google.com/drive/folders/1kf3AXpKbwbZuQhcVSiaMzCiaSrWTdO7i?usp=sharing).
- The datasets used in the experiments are: `nyt-fine`, `Reddit`, `DBPedia`

#### Get Data
- [**DSMLP Users**]: For the 3 datasets provided, convenient Linux commands to download and get the data are provided in the [documentation of raw data](data/raw/). Please run the commands in the **repository root directory**.
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

### Commands
[**DSMLP Users**]: 
- The `test` target could be easily run as `python run.py test`.
- The `experiment` target could be run as `python run.py exp <dataset>`.
- When prompted from the prompt, insert values by manually inspecting the plots generated in `artifacts/`.

The main script is located in the root directory. It supports 3 targets:
- `test`: Run the test data. All other flags are ignored.
- `experiment` (or `exp`) [default]: Perform one vanilla run.

The full command is:
```
python run.py [-h] target [-d DATA]

required: target {test,experiment,exp}
  run target. Default experiment; if test is selected, ignore all other flags.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  data path, required for non-testdata
```
**Note**: Due to time constraints and container constraints, the short experiments are chosen to run fast, which means performance is not guaranteed.


### Code File Outline
```
Incomplete-Text-Classification/
├── run.py                           <- main run script
├── data/                            <- all data files
│   ├── raw                          <- raw files (after download)
│   │   ├── nyt-fine
│   │   |   └── df.pkl               <- required DataFrame pickle file
│   |   └── ...
│   └── processed/                   <- processed files (after preprocessing)
├── src/                             <- source code library
│   ├── data.py                      <- data class definition
│   ├── base_modules.py              <- modules as basic components
│   ├── baseline_model.py            <- simple baseline model
│   ├── evaluation.py                <- evaluation methods
│   └── util.py                      <- other utility functions
└── test/                            <- test target data
```

---
### Citations

#### Word2Vec
```
@article{word2vec,
    title={Efficient estimation of word representations in vector space},
    author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
    journal={arXiv preprint arXiv:1301.3781},
    year={2013}
}
```