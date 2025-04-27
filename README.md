# EEG Analysis using UFAL's EMMT dataset

This repository contains the code used to analyze the EEG data using the UFAL's EMMT dataset


## Data

The data is accessed directly from [UFAL's EMMT GitHub repository](https://github.com/ufal/eyetracked-multi-modal-translation) using the GitHub API. No local data files are needed!

## Features

- Fetches data directly from GitHub using the API or from local directory
- Visualizes EEG, audio, and gaze data
- Applies various preprocessing techniques to EEG data
- Filter data by participant, sentence, or category

## Environment Variables

The application can be configured using the following environment variables:

```bash
# Use local files (default)
FILE_LOADER_TYPE="local"
DATA_PATH="./ufal_emmt"

# OR use GitHub API
FILE_LOADER_TYPE="github"
DATA_PATH="https://api.github.com/repos/karelvlk/ufal_emmt"
```

## Run streamlit app for visualization

```bash
poetry install
poetry run streamlit run app.py
```

## Requirements

- Python 3.11+
- Dependencies as specified in pyproject.toml (automatically installed by Poetry)


