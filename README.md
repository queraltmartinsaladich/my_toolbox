# My ML Toolbox 

A streamlined, high-level Python framework designed to accelerate the Machine Learning lifecycle—from raw data ingestion and debugging to advanced text similarity and model evaluation.

## 🚀 Features

* **Automated Debugging**: Catch nulls, empty strings, and data leakage before training.
* **Text Preprocessing**: Clean HTML, normalize Unicode, and handle multi-format I/O (CSV, JSON, TXT).
* **Modular Modeling**: Decoupled classes for Classification, Regression, and Clustering.
* **Similarity Suite**: Generate t-SNE projections, Dendrograms, and Heatmaps with one command.
* **Professional Viz**: Blue-themed confusion matrices and polished ROC curves.

---

## 📦 Installation

```pip install git+https://github.com/queraltmartinsaladich/my_toolbox.git```

```from my_toolbox import *```

---

## 📂 Project Structure

my_toolbox_project/
├── my_toolbox/                # Main package directory
│   ├── __init__.py            # Exports the public API
│   ├── data_utils.py          # EDA and human-readable stats
│   ├── debug_utils.py         # Pipeline safety and data cleaning
│   ├── metric_utils.py        # Evaluation reports and feature importance
│   ├── model_utils.py         # Class-based models (Clustering, Classif., etc.)
│   ├── plot_utils.py          # Visualization logic (t-SNE, Heatmaps, ROC)
│   └── text_utils.py          # Preprocessor class and Label Encoding
├── .gitignore                 # Excludes __pycache__, models/, and results/
├── pyproject.toml             # Build system and dependencies
└── README.md                  # Documentation and usage guide

### File Responsibilities

> `debug_utils.py`: It catches nulls, empty strings, and data leakage before you hit model.fit().

> `data_utils.py`: It calculates word counts, class balances, and length correlations.

> `text_utils.py`: Handles Unicode normalization, HTML stripping, and recursive cleaning of JSON/Lists.

> `model_utils.py`: Decoupled so you can use your own embeddings or the built-in Vectorization class.

> `plot_utils.py`: Centralizes all aesthetic choices (color palettes, figure sizes, and axis formatting).

> `metric_utils.py`: It imports functions from plot_utils and model_utils to generate combined artifacts.