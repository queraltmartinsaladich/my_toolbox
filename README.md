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

### File Responsibilities

**- LOADERS**

`image_loader.py`: A specialized PyTorch Dataset that standardizes image ingestion. It handles conversion from HWC (OpenCV) to CHW (Torch), standardizes color spaces (RGB/Grayscale), and ensures that masks and images are transformed identically during augmentation.

`table_loader.py`: Designed for neural networks processing tabular data. It handles the mapping of continuous and categorical features from Pandas DataFrames into float tensors.

`text_loader.py`: A wrapper for Hugging Face tokenizers. It transforms raw strings into the token IDs and attention masks required for Transformer-based models.

**- CORE MODULES**

`cluster_utils.py`: It identifies optimal cluster counts via Elbow/Silhouette analysis and interprets clusters by extracting significant keywords or centroids.

`data_utils.py`: It provides human-readable statistics, such as class balance reports, word count distributions, and feature correlations.

`debug_utils.py`: It identifies and sanitizes data corruption (nulls, infinite values, empty strings) and checks for data leakage between train/test splits.

`deep_learning.py`: It encapsulates the DeepPipeline class, managing the training loops, GPU/CUDA device placement, and the interface for Hugging Face AutoModels.

`loading_utils.py`: It standardizes the ingestion of .csv, .json, and .txt files to ensure they are presented to the loaders in a consistent format.

`metric_utils.py`: It calculates mathematical performance (F1, IoU, MAE) and aggregates them with feature importance data to generate combined JSON and text artifacts.

model_utils.py`: It houses the Classification, Regression, and Similarity classes. It is decoupled to allow seamless switching between TF-IDF vectorization and custom embeddings.

`other_utils.py`: Contains execution timers (@time_it), hardware/system auditors, and YAML/JSON configuration loaders.

`plot_utils.py`: Centralizes all visual logic, from ROC curves and Confusion Matrices to t-SNE projections and segmentation side-by-side comparisons.

`text_utils.py`: Manages the "dirty work" of NLP, including Unicode normalization, HTML stripping, stop-word removal, and label encoding.

---

## 🚀 Usage Guide

The toolbox is split into two primary execution engines depending on your data type and hardware availability.

### 1. Classical ML Pipeline (main_classic.py)

_Best for: Tabular data, small-to-medium text datasets, and quick statistical analysis._
_Hardware: Optimized for CPU._

- Args for main_classic.py:

> data_path: Path to CSV, JSON, or TXT file <br>
> task: Classification 'clf', Regression 'reg', Similarity 'sim', K-means clustering 'km' <br>
> text_col: Column name for text <br>
> label_col: Column name for target <br>
> subset: Process only N (subset) samples <br>

**Text Classification (TF-IDF + Logistic Regression):**

`python main_classic.py --path data.csv --task clf --target label --text_col review_text --vectorize`

**Tabular Regression (Ridge):**

`python main_classic.py --path housing.csv --task reg --target price`

**Similarity & Clustering:**

`python main_classic.py --path documents.json --task sim --text_col content`

#### 2. Deep Learning Pipeline (main_deep.py)

_Best for: Computer Vision, Transformer-based NLP, and high-dimensional patterns._
_Hardware: Optimized for GPU._

- Args for main_deep.py:

> data_path: Path to CSV, JSON, or TXT file <br>
> type: 'image' or 'text' <br>
> task: Classification 'clf', Regression 'reg', Segmentation 'seg' <br>
> model: Hugging Face string or local model path <br>
> epochs: Number of epochs <br>
> lr: Learning rate <br>
> batch: Number of samples in batch <br>
> label: Column name for target <br>
> subset: Process only N (subset) samples <br>

**Image Segmentation (U-Net/ResNet):**

`python main_deep.py --path manifest.csv --type image --task seg --model resnet50 --batch 4`

**Transformer Text Classification (BERT/DistilBERT):**

`python main_deep.py --path tweets.csv --type text --task clf --model distilbert-base-uncased --epochs 3`

**Custom Torch Models:**

`python main_deep.py --path custom_data.csv --type image --task clf --model ./models/my_custom_architecture.pt`

---

### 🛠 Project Architecture Flow

Input: Provide a file path (CSV/JSON/TXT) containing your features.

Loading: main_(deep/classic).py calls ToolboxDataLoader, selecting the correct specialized loader (image, text, or table).

Validation: debug_utils.py audits the data for nulls or leakage before tensors hit the model.

Modeling: main_classic.py uses class-based models from model_utils.py. main_deep.py uses the deep learning engine from deep_learning.py.

Output: metric_utils.py aggregates scores while plot_utils.py generates visual artifacts (ROC, Heatmaps, IoU samples) in the /results directory.