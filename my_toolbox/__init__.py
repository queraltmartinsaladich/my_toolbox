"""
My ML Toolbox: A streamlined framework for NLP, Clustering, and Evaluation.
"""

# 1. Core Data & Text Transformation (text_utils.py)
from .text_utils import (
    Preprocessor, 
    prepare_labels)

# 2. Pipeline Safety & Cleaning (debug_utils.py)
from .debug_utils import (
    debug_dataframe, 
    check_text_duplicates, 
    check_tokenization_viability)

# 3. Data Health & EDA (data_utils.py)
from .data_utils import (
    full_data_inspection, 
    get_length_stats_by_label,
    inspect_structure,
    inspect_text_quality)

# 4. Modeling & Vectorization (model_utils.py)
from .model_utils import (
    Vectorization,
    Classification,
    Regression,
    Similarity,
    Clustering,
    save_model,
    load_model)

# 5. Evaluation & Explainability (metric_utils.py)
from .metric_utils import (
    full_classification_metrics, 
    full_similarity_report,
    get_top_features)

# 6. Low-level Visuals (plot_utils.py)
from .plot_utils import (
    plot_confusion_matrix, 
    plot_roc_curve,
    plot_similarity_heatmap,
    plot_text_tsne,
    plot_text_dendrogram)

# --- Versioning ---
__version__ = "1.0.0"
__author__ = "Queralt Martín-Saladich"

# --- Public API ---
__all__ = [
    # Text & Labels
    "Preprocessor", 
    "prepare_labels",
    
    # Debugging
    "debug_dataframe", 
    "check_text_duplicates", 
    "check_tokenization_viability",
    
    # Inspection
    "full_data_inspection", 
    "get_length_stats_by_label",
    "inspect_structure",
    "inspect_text_quality",
    
    # Modeling
    "Vectorization",
    "Classification",
    "Regression",
    "Similarity",
    "Clustering",
    "save_model",
    "load_model",
    
    # Metrics
    "full_classification_metrics", 
    "full_similarity_report",
    "get_top_features",
    
    # Plotting
    "plot_confusion_matrix", 
    "plot_roc_curve",
    "plot_similarity_heatmap",
    "plot_text_tsne",
    "plot_text_dendrogram"]