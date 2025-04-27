# EEG Analysis using UFAL's EMMT dataset

This repository contains the code used to analyze the EEG data using the UFAL's EMMT dataset, fetching data directly from GitHub via the API.

## Data

The data is accessed directly from [UFAL's EMMT GitHub repository](https://github.com/ufal/eyetracked-multi-modal-translation) using the GitHub API. No local data files are needed!

## Features

- Fetches data directly from GitHub using the API
- Visualizes EEG, audio, and gaze data
- Applies various preprocessing techniques to EEG data
- Filter data by participant, sentence, or category

## Run streamlit app for visualization

```bash
poetry install
poetry run streamlit run app.py
```

## Requirements

- Python 3.11+
- Dependencies as specified in pyproject.toml (automatically installed by Poetry)


