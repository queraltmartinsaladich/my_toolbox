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
```
my_toolbox_project/
├── my_toolbox/                # Main package directory
│   ├── loaders/               # Loader directory for different input types
│   │   ├── image_loader.py    # Image DataLoader
│   │   ├── table_loader.py    # Table DataLoader
│   │   └── text_loader.py     # Text DataLoader
│   ├── __init__.py            # Exports the public API
│   ├── cluster_utils.py       # Clusterization functions
│   ├── data_utils.py          # EDA and human-readable stats
│   ├── debug_utils.py         # Pipeline safety and data cleaning
│   ├── deep_learning.py       # Deep learning models module
│   ├── loading_utils.py       # Loading dataset functions
│   ├── metric_utils.py        # Evaluation reports and feature importance
│   ├── model_utils.py         # Class-based models (Clustering, Classif., etc.)
│   ├── other_utils.py         # Auditors, time decorators...
│   ├── plot_utils.py          # Visualization logic (t-SNE, Heatmaps, ROC)
│   └── text_utils.py          # Preprocessor class and Label Encoding
├── .gitignore                 # Excludes __pycache__, models/, and results/
├── main_classic.py            # Full pipeline for classic ML
├── main_deep.py               # Full pipeline for DL-ML
├── pyproject.toml             # Build system and dependencies
└── README.md                  # Documentation and usage guide
```

#### File Responsibilities

`debug_utils.py`: It catches nulls, empty strings, and data leakage before you hit model.fit().

`data_utils.py`: It calculates word counts, class balances, and length correlations.

`text_utils.py`: Handles Unicode normalization, HTML stripping, and recursive cleaning of JSON/Lists.

`model_utils.py`: Decoupled so you can use your own embeddings or the built-in Vectorization class.

`plot_utils.py`: Centralizes all aesthetic choices (color palettes, figure sizes, and axis formatting).

`metric_utils.py`: It imports functions from plot_utils and model_utils to generate combined artifacts.

---

## 💡 How to use (see main.py example)

1. For CSV/JSON Training (Classification) --> This will run the debug pipeline, clean the text, train the model, and save a full report (images and JSON) in the /results folder.

> `python main_classic.py --path data.csv --task clf --target label --text_col review_text --vectorize`

2. For Raw Text Discovery (Clustering) --> This will read each line of your .txt file as a document, group them into 5 themes, and print the top words for each.

> `python main.py --path data/raw_feedback.txt --task km`

3. For Similarity Analysis (Duplicates) --> This is great for cleaning a dataset. It will find rows that are almost identical.

> `python main.py --path data/corpus.csv --task sim --text content`

🚀 Usage Guide
The toolbox is split into two primary execution engines depending on your data type and hardware availability.

1. Classical ML Pipeline (main_classic.py)

Best for: Tabular data, small-to-medium text datasets, and quick statistical analysis.

Hardware: Optimized for CPU.

Text Classification (TF-IDF + Logistic Regression):

Bash
python main_classic.py --path data.csv --task clf --target label --text_col review_text --vectorize
Tabular Regression (Ridge):

Bash
python main_classic.py --path housing.csv --task reg --target price
Similarity & Clustering:

Bash
python main_classic.py --path documents.json --task sim --text_col content
2. Deep Learning Pipeline (main_deep.py)

Best for: Computer Vision, Transformer-based NLP, and high-dimensional patterns.

Hardware: Optimized for GPU (CUDA).

Image Segmentation (U-Net/ResNet):

Bash
python main_deep.py --path manifest.csv --type image --task segmentation --model resnet50 --batch 4
Transformer Text Classification (BERT/DistilBERT):

Bash
python main_deep.py --path tweets.csv --type text --task classification --model distilbert-base-uncased --epochs 3

Custom Torch Models:

Bash
python main_deep.py --path custom_data.csv --type image --task classification --model ./models/my_custom_architecture.pt

---

### 🛠 Project Architecture Flow

Input: Provide a Manifest (CSV/JSON) containing paths to raw files or tabular features.

Loading: main_*.py calls ToolboxDataLoader, selecting the correct specialized loader (image, text, or table).

Validation: debug_utils.py audits the data for nulls or leakage before tensors hit the model.

Modeling: * main_classic uses class-based models from model_utils.py.

main_deep uses the DeepPipeline engine from deep_learning.py.

Output: metric_utils.py aggregates scores while plot_utils.py generates visual artifacts (ROC, Heatmaps, IoU samples) in the /results directory.