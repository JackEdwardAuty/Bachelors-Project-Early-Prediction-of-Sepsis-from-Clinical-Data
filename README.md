This is a project I completed for my bachelors thesis, I utilised bespoke Python code implementing the **PyTorch** open-source machine learning library to achieve in-depth data analysis on a large medical dataset.


The dataset has been pruned from this repository due to GitHub file size restraints, and [can be found here](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis).


Original 43 commits could not preserved in the transfer from GitLab because GitHub does not support the large files of the dataset that were part of many of those commits. This commit history can instead be viewed by the curious at [the original host location](https://gitlab.com/JackAuty/comp3931-individual-project/-/commits/master?ref\_type=HEADS).



# Early Prediction of Sepsis from Clinical Data

This repository contains my Bachelors project on building a machine‑learning pipeline for **early prediction of sepsis from ICU clinical data**. The goal is to predict sepsis several hours before clinical recognition, using routinely collected vital signs, labs and demographics, in line with recent sepsis‑prediction challenges and literature.

Sepsis is a life‑threatening organ dysfunction caused by a dysregulated host response to infection, and early detection is critical for improving outcomes. Machine‑learning models have shown promise in identifying subtle patterns in electronic health record (EHR) time‑series that precede sepsis onset.

## Project structure
  <br>.
  <br>├── bachelors-report/ # Final year report and relevant documents
  <br>│ ├── assets/ # Templates and other files used
  <br>│ ├── Intermediate_Report.odt
  <br>│ └── Early_Prediction_of_Sepsis_from_Clinical_Data-Jack_Auty_Bachelors_Report.pdf # Final Report Submitted
  <br>├── project-dataset/
  <br>│ ├── PSVs-not-included.txt
  <br>│ ├── raw/ # Original clinical time-series (not committed if sensitive/filesize restraints)
  <br>│ ├── processed/ # Cleaned, windowed data ready for modelling
  <br>│ └── metadata/ # Feature lists, variable descriptions, cohort definitions
  <br>├── project-source
  <br>│ ├── assemble.py # DataLoader class: load & preprocess time-series
  <br>│ ├── classes.py # ModelTrainer class: train & cross-validate models
  <br>│ ├── neuralnetwork.py # Evaluator class: metrics, ROC/PR plots, SHAP
  <br>│ └── model.pt
  <br>├── uml/ 
  <br>│ ├── class_diagram.png
  <br>│ └── sequence_diagram.png
  <br>├── visualisations/
  <br>│ ├── exploratory/
  <br>│ │ ├── exploring-feature-distribution/
  <br>│ │ └── exploring-feature-distribution-with-histograms/
  <br>│ └── showcase/ # Most clinically releveant visualisations and ones used in report
  <br>├── requirements.txt # Python dependencies
  <br>└── README.md

---

## Architecture

The architecture follows three main components:

- **DataLoader** – loads raw clinical time‑series, aligns measurements to hourly grid, handles missing values, and creates fixed‑length windows (e.g. 6‑hour look‑back) for prediction.
- **ModelTrainer** – trains ML models (e.g. XGBoost / tree‑based models) with cross‑validation and hyperparameter tuning.  
- **Evaluator** – computes metrics (AUROC, AUPRC, accuracy, sensitivity, specificity) and produces ROC/PR curves and feature‑importance or SHAP plots.

UML diagrams in `uml/` document how these pieces interact and help communicate the design to both technical and clinical audiences.

![Class Diagram](https://github.com/JackEdwardAuty/Bachelors-Project-Early-Prediction-of-Sepsis-from-Clinical-Data/blob/main/uml/class_diagram.png)
![Sequence Diagram](https://github.com/JackEdwardAuty/Bachelors-Project-Early-Prediction-of-Sepsis-from-Clinical-Data/blob/main/uml/sequence_diagram.png)

---

## Data

![](visualisations/showcase/Figure_5:_HR_vs_SBP_Scatterplot_with_Density_Histograms.png)

This project is designed for structured ICU time‑series data similar to PhysioNet / MIMIC‑III sepsis datasets: hourly measurements of vital signs, labs and demographics with binary labels indicating sepsis onset.

Because access to real patient data is restricted, **no raw data is committed to this repo**. To reproduce results you will need:

1. Access to an appropriate sepsis dataset (e.g. [PhysioNet sepsis challenge data](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis) or a local EHR extract).  
2. To modify `data_config.yaml` or equivalent to point to your local data paths and variable names.

Where possible, `project-dataset/metadata/` contains feature lists and descriptions to help map your dataset into the expected format.

---

## Methods

High‑level pipeline:

1. **Cohort selection** – select ICU stays that meet basic inclusion criteria and define sepsis onset times using Sepsis‑3 or challenge‑provided labels.
2. **Time‑series preprocessing** – resample to an hourly grid, align multiple signals, forward‑fill or impute missing values, and normalise numeric features.
3. **Windowing** – for each hour, construct a fixed‑length look‑back window (e.g. last 6 hours) and label the window as positive if sepsis will occur within the prediction horizon.  
4. **Modelling** – train ML models (e.g. XGBoost, random forest, logistic regression) to classify windows as high‑ or low‑risk.  
5. **Evaluation** – evaluate using AUROC and AUPRC on a held‑out test set, with emphasis on early‑warning performance in an imbalanced setting.

Implementation details and results are documented in the notebooks and in the written dissertation (not included here).

---

## Quick‑start

This quick‑start is for someone who wants to run the full pipeline on their own machine using a compatible dataset.

### 1. Clone the repository

  `git clone https://github.com/JackEdwardAuty/Bachelors-Project-Early-Prediction-of-Sepsis-from-Clinical-Data.git`
<br>`cd Bachelors-Project-Early-Prediction-of-Sepsis-from-Clinical-Data`


### 2. Set up the environment

Using `venv` (example with Python ≥ 3.9):

<br>`python -m venv .venv`
<br>`source .venv/bin/activate` # Windows: `.venv\Scripts\activate`
<br>`pip install --upgrade pip`
<br>`pip install -r requirements.txt`

If you use conda, create an environment with the packages listed in `requirements.txt` instead.

### 3. Prepare the data

1. Obtain a suitable sepsis dataset (e.g. [PhysioNet 2019 sepsis challenge](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis) or similar ICU time‑series data).
2. Place files under `project-dataset/raw/` in the expected format (documented in `project-dataset/metadata/README.md` or comments in `dataloader.py`).  
3. If needed, edit configuration variables in `project-source/dataloader.py` or a config file (e.g. column names, time column, label column, prediction horizon).

### 4. Run preprocessing

From the repo root:

  `python -m project-source.dataloader`

Typical actions:

- Load raw patient‑level files.  
- Resample and align to hourly time‑steps.  
- Handle missing values and basic feature engineering.  
- Save processed data to `project-datasetprocessed/`.

Check the console output or logs to confirm the number of patients and windows created.

### 5. Train a baseline model

  `python -m project-source.trainer`

This script should:

- Load `project-datasetprocessed/` datasets.  
- Split data into train/validation/test sets.  
- Train a baseline model (e.g. XGBoost) with default hyperparameters.  
- Save the model to `models/` and basic metrics to `results/`.

You can adjust model choice and hyperparameters via command‑line flags or a config file (documented at the top of `trainer.py`).

### 6. Evaluate and inspect results

`python -m project-source.evaluate`

This will typically:

- Load trained models and test data.  
- Compute AUROC, AUPRC, confusion matrices, and calibration curves.  
- Save plots (ROC/PR curves, feature importances or SHAP summaries) into `figures/`.

Open the figures and metrics to assess performance and compare against benchmarks from the sepsis‑prediction literature.

---

## Next steps / reassessment ideas

If you are revisiting this project, potential improvements include:

- Trying alternative models (e.g. deep time‑series models such as LSTM, GRU or Transformer‑based architectures).  
- Adding **calibration** and decision‑curve analysis to better understand clinical usefulness.  
- Performing external validation or cross‑dataset evaluation, which recent papers stress as necessary for generalisable sepsis models. 
- Improving documentation and unit tests around `DataLoader`, `ModelTrainer`, and `Evaluator` to make the pipeline more robust and reusable.

These steps will both strengthen the scientific quality of the project and make it more compelling as a portfolio piece for data and ML roles.
