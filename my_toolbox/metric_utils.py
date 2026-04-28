import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .plot_utils import plot_confusion_matrix, plot_roc_curve, plot_similarity_heatmap, plot_text_tsne, plot_text_dendrogram

logger = logging.getLogger(__name__)

def get_top_features(vectorizer, model, n=10):
    """Returns the words that most strongly predict the classes."""
    feature_names = vectorizer.get_feature_names_out()
    if model.coef_.shape[0] == 1:
        # Binary Classification
        coefs = model.coef_[0]
        top_indices = np.argsort(coefs)[-n:]
        results = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_indices],
            'Coefficient': [coefs[i] for i in top_indices]
        }).sort_values(by='Coefficient', ascending=False)
    else:
        # Multi-class: This returns a list of dataframes for each class
        results = {}
        for i, class_label in enumerate(model.classes_):
            coefs = model.coef_[i]
            top_indices = np.argsort(coefs)[-n:]
            results[class_label] = pd.DataFrame({
                'Feature': [feature_names[j] for j in top_indices],
                'Coefficient': [coefs[j] for j in top_indices]
            }).sort_values(by='Coefficient', ascending=False)
    return results

def full_classification_metrics(y_true, y_pred, y_probs=None, classes=None, output_dir="results", analysis="test"):
    """
    Generates a full evaluation report: metrics (JSON/TXT), 
    CSV confusion matrix, and plots (ROC & Heatmap).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Classification Report
    report_text = classification_report(y_true, y_pred, target_names=classes)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    print("\nClassification Report:\n", report_text)

    with open(os.path.join(output_dir, f"report_{analysis}.txt"), "w") as f:
        f.write(report_text)
    with open(os.path.join(output_dir, f"metrics_{analysis}.json"), "w") as f:
        json.dump(report_dict, f, indent=4)
        
    # 2. Confusion Matrix (Plot)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        y_true, 
        y_pred, 
        classes=classes if classes else [str(i) for i in range(len(cm))], 
        save_path=os.path.join(output_dir, f"confusion_matrix_{analysis}.png"))
    
    # 3. ROC Curve
    if y_probs is not None:
        auc_score = plot_roc_curve(
            y_true, 
            y_probs, 
            os.path.join(output_dir, f"roc_curve_{analysis}.png"))
        logger.info(f"ROC AUC Score: {auc_score:.4f}")
    
    logger.info(f"Evaluation saved to {output_dir}/")
    return report_dict

def full_similarity_report(matrix, labels, output_dir="results", analysis="test"):
    """Runs all similarity analyses and saves them to a folder."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating similarity reports in {output_dir}...")
    sim_matrix = plot_similarity_heatmap(matrix, labels, output_dir, analysis)
    plot_text_tsne(matrix, labels, output_dir, analysis)
    plot_text_dendrogram(matrix, labels, output_dir, analysis)
    df_sim = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    df_sim.to_csv(os.path.join(output_dir, f"similarity_scores_{analysis}.csv"))
    print(f"Evaluation saved to {output_dir}/")
    return sim_matrix