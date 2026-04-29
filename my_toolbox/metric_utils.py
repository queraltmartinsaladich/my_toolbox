import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

from .plot_utils import * 

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
            'Coefficient': [coefs[i] for i in top_indices]}).sort_values(by='Coefficient', ascending=False)
    
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

def compare_models(results_dict: dict):
    """
    Accepts a dict: {'ModelName': {'f1': 0.9, 'accuracy': 0.8}, ...}
    Returns a sorted comparison table.
    """
    df = pd.DataFrame(results_dict).T
    df = df.sort_values(by=df.columns[0], ascending=False) # Sort by first metric
    print("\n--- Model Comparison Table ---")
    print(df)
    return df

def full_regression_metrics(y_true, y_pred, output_dir="results", analysis="test"):
    """
    Generates MAE, RMSE, and R2 scores, saves to JSON, 
    and generates an Actual vs. Predicted scatter plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }
    
    logger.info(f"Regression Metrics: MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # 1. Save JSON report
    with open(os.path.join(output_dir, f"reg_metrics_{analysis}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # 2. Generate Scatter Plot
    plot_path = os.path.join(output_dir, f"regression_analysis_{analysis}.png")
    plot_regression_results(y_true, y_pred, save_path=plot_path)
    
    return metrics

def segmentation_metrics(y_true, y_pred, image=None, output_dir="results", analysis="test", smooth=1e-6):
    """
    Calculates IoU and Dice Coefficient. 
    If an image is provided, generates a side-by-side visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten tensors/arrays for calculation
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    metrics = {"iou": float(iou), "dice": float(dice)}
    logger.info(f"Segmentation Metrics: IoU: {iou:.4f}, Dice: {dice:.4f}")

    # 1. Save JSON report
    with open(os.path.join(output_dir, f"seg_metrics_{analysis}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 2. Generate Visual Comparison (if image is passed)
    if image is not None:
        plot_path = os.path.join(output_dir, f"segmentation_sample_{analysis}.png")
        plot_segmentation_sample(image, y_true, y_pred, save_path=plot_path)

    return metrics

def save_training_history(history: dict, output_dir="results"):
    """
    Saves the loss/accuracy history from a NeuralPipeline.
    Expects history = {'train_loss': [...], 'val_loss': [...]}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    
    # Plot curves
    plot_loss_curves(history, save_path=os.path.join(output_dir, "loss_curves.png"))
    logger.info(f"Training history and plots saved to {output_dir}")