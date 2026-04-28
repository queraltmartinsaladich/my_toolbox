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
```

#### File Responsibilities

`debug_utils.py`: It catches nulls, empty strings, and data leakage before you hit model.fit().

`data_utils.py`: It calculates word counts, class balances, and length correlations.

`text_utils.py`: Handles Unicode normalization, HTML stripping, and recursive cleaning of JSON/Lists.

`model_utils.py`: Decoupled so you can use your own embeddings or the built-in Vectorization class.

`plot_utils.py`: Centralizes all aesthetic choices (color palettes, figure sizes, and axis formatting).

`metric_utils.py`: It imports functions from plot_utils and model_utils to generate combined artifacts.

---

## 💡 How to use (see main.py example below)

1. For CSV/JSON Training (Classification) --> This will run the debug pipeline, clean the text, train the model, and save a full report (images and JSON) in the /results folder.

> `python main.py --path data/train.csv --task clf --text review_body --label rating`

2. For Raw Text Discovery (Clustering) --> This will read each line of your .txt file as a document, group them into 5 themes, and print the top words for each.

> `python main.py --path data/raw_feedback.txt --task km`

3. For Similarity Analysis (Duplicates) --> This is great for cleaning a dataset. It will find rows that are almost identical.

> `python main.py --path data/corpus.csv --task sim --text content`

### 📌 Example main.py

```
import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge

# Import from your installed toolbox
from my_toolbox import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(data_path, task, text_col="text", label_col="label", subset=None):
    try:
        # 1. Initialize Preprocessor and Load Data
        prep = Preprocessor()
        raw_input = prep.load_any_file(data_path)
        
        X, y = [], None
        classes = None

        # --- BRANCH A: TABULAR DATA (CSV / JSON) ---
        if isinstance(raw_input, pd.DataFrame):
            logger.info(f"Processing tabular data from {data_path}")
            
            # Debug and Inspect (Human-readable EDA)
            df = debug_dataframe(raw_input, text_col, label_col)
            
            if subset:
                logger.info(f"Sampling subset of {subset} rows.")
                df = df.sample(n=min(subset, len(df)), random_state=42)
            
            full_data_inspection(df, text_col, label_col)

            # Clean text column
            df[text_col] = prep.process(df[text_col])
            X = df[text_col].tolist()
            
            if label_col in df.columns:
                y = df[label_col].values

        # --- BRANCH B: RAW TEXT (.txt) ---
        elif isinstance(raw_input, str):
            logger.info(f"Processing raw text file from {data_path}")
            # We treat lines as separate documents for tasks like Clustering/Similarity
            lines = [prep.clean_text(line) for line in raw_input.split('\n') if line.strip()]
            X = lines
            y = None
            logger.info(f"Extracted {len(X)} lines/documents from file.")

        else:
            logger.error("Unsupported data type loaded.")
            return

        # 3. Initialize Shared Tools
        vec = Vectorization()
        os.makedirs("results", exist_ok=True)

        # --- TASK EXECUTION ---

        if task == "clf":
            if y is None: raise ValueError("Classification (clf) requires a label column.")
            
            y_enc, enc = prepare_labels(y)
            classes = list(enc.classes_)
            
            Xtr, Xts, ytr, yts = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=3)
            
            Xtr_vec = vec.fit_transform(Xtr)
            Xts_vec = vec.transform(Xts)
            
            model = Classification(LogisticRegression(class_weight='balanced'))
            model.train(Xtr_vec, ytr, cv=5)
            
            # Post-modeling: Explain and Evaluate
            get_top_features(vec, model.model, n=15)
            ypred = model.predict(Xts_vec)
            full_classification_metrics(yts, ypred, classes=classes, analysis="main_run")

        elif task == "reg":
            if y is None: raise ValueError("Regression (reg) requires a numeric target column.")
            Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=3)
            
            Xtr_vec = vec.fit_transform(Xtr)
            Xts_vec = vec.transform(Xts)
            
            model = Regression(Ridge())
            model.train(Xtr_vec, ytr, cv=5)
            ypred = model.predict(Xts_vec)
            logger.info(f"Regression Samples: {ypred[:5]}")

        elif task == "sim":
            X_vec = vec.fit_transform(X)
            sim_engine = Similarity()
            
            # 1. Find internal duplicates
            dupes = sim_engine.find_duplicates(X, vectorizer=vec, threshold=0.9)
            logger.info(f"Found {len(dupes)} near-duplicate pairs.")
            
            # 2. Visual Similarity Report (t-SNE, Dendrogram, Heatmap)
            full_similarity_report(X_vec, labels=classes if classes else X[:10], analysis="main_run")

        elif task == "km":
            X_vec = vec.fit_transform(X)
            model = Clustering(n_clusters=5)
            cluster_labels = model.train(X_vec)
            
            # Explain the clusters
            model.get_top_terms_per_cluster(vec)
            
            # Save assignments if tabular
            if isinstance(raw_input, pd.DataFrame):
                raw_input['cluster'] = cluster_labels
                raw_input.to_csv("results/clustered_output.csv", index=False)
                logger.info("Clusters saved to results/clustered_output.csv")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to CSV, JSON, or TXT file")
    parser.add_argument('--task', type=str, choices=['clf', 'reg', 'sim', 'km'], required=True)
    parser.add_argument('--text', type=str, default='text', help="Column name for text")
    parser.add_argument('--label', type=str, default='label', help="Column name for target")
    parser.add_argument('--subset', type=int, default=None, help="Process only N samples")
    
    args = parser.parse_args()
    run_pipeline(args.path, args.task, args.text, args.label, args.subset)
```