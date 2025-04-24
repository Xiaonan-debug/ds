# Text Classification Project

This repository provides scripts and a notebook for training and evaluating text classification models using both transformer-based (BERT) and classical machine learning approaches. It also includes data extraction and cleaning utilities.

## Repository Structure

```
├── bert.ipynb            # Jupyter notebook example for BERT training and evaluation
├── bert.py               # Python script: BERT training with k-fold CV and hyperparameter tuning
├── cls.py                # Python script: Classical ML training with TF-IDF and various classifiers
├── evaluate.py           # Python script: Evaluate ROC AUC for classical classifiers
├── data.py               # Python script: Extracts and processes JSON responses into CSV
├── clean.py              # Python script: Filters out entries with 'Unknown' labels from data.csv
├── data_filtered.csv     # (Input/Output) Filtered dataset with "text" and "label" columns
└── requirements.txt      # Required Python packages
```

## Prerequisites

- Python 3.7 or higher
- GPU recommended for BERT training

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Workflow

### 1. Data Extraction (`data.py`)

Extracts text and judgment labels from `all_data_judge_res.json` and outputs `data.csv` with columns:
- `text`: response text
- `label`: binary labels (0 for Satisfied, 1 for Not Satisfied, or "Unknown")
- `keys`: model identifiers

```bash
python data.py
```

### 2. Data Cleaning (`clean.py`)

Removes entries with `"Unknown"` labels from `data.csv` and saves the result to `data_filtered.csv`.

```bash
python clean.py
```

## Model Training and Evaluation

### 3. BERT Training and Evaluation (`bert.py`)

Performs hyperparameter tuning on the first fold and then runs k-fold cross-validation with the best settings.

```bash
python bert.py
```

- **Output directory**: `./bert_results/`
  - CSV metrics: `bert_metrics.csv`
  - Hyperparameter tuning plot: `hyperparameter_tuning.png`
  - Fold results plot: `fold_results.png`

### 4. Classical ML Training (`cls.py`)

Performs grid search hyperparameter tuning on TF-IDF + KNN by default (configurable), then evaluates with k-fold CV.

```bash
python cls.py
```

- **Outputs**:
  - CSV with model metrics: `model_metrics.csv`
  - Metrics table image: `model_metrics_table.png`

### 5. ROC AUC Evaluation (`evaluate.py`)

Evaluates multiple classical classifiers with k-fold cross-validation to compute ROC curves and AUC scores.

```bash
python evaluate.py
```

- **Outputs**:
  - ROC curves plot: `roc_curves.png`
  - AUC scores bar chart: `auc_scores.png`
  - CSV with AUC scores: `auc_scores.csv`

## Jupyter Notebook (`bert.ipynb`)

An interactive example demonstrating BERT training and evaluation. Open with:

```bash
jupyter notebook bert.ipynb
```

## Requirements File

```text
pandas
numpy
matplotlib
scikit-learn
torch
transformers
datasets
seaborn
``` 

*(Standard Python library `json` and `re` are used in data extraction and cleaning scripts.)*

## License

This project is released under the MIT License. Feel free to reuse and modify.

