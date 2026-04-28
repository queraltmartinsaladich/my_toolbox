import logging
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)

def plot_roc_curve(y_true, y_probs, save_path):
    """Generates and saves the ROC curve plot"""
    fpr, tpr, th = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Generates a blue-themed confusion matrix with counts and percentages.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype('float') / cm.sum() * 100
    labels = (np.array(["{0:d}\n({1:.1f}%)".format(count, perc)
              for count, perc in zip(cm.flatten(), cm_perc.flatten())]
             )).reshape(cm.shape)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt="", 
        cmap='Blues', 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=True)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return cm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_similarity_heatmap(matrix, labels, output_dir="results", analysis="test"):
    """Generates a Cosine Similarity Heatmap."""
    sim_matrix = cosine_similarity(matrix)
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, cmap="YlGnBu", annot=False)
    plt.title("Document Similarity Heatmap (Cosine)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_{analysis}.png"))
    plt.close()
    return sim_matrix

def plot_text_tsne(matrix, labels, output_dir="results", analysis="test"):
    """Generates a 2D t-SNE scatter plot of document embeddings."""
    data = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    perp = min(30, max(5, len(labels) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, c='royalblue', edgecolors='k')
    for i, label in enumerate(labels):
        plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=9, alpha=0.8)
    plt.title("t-SNE Text Cluster Projection")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"tsne_{analysis}.png"))
    plt.close()

def plot_text_dendrogram(matrix, labels, output_dir="results", analysis="test"):
    """Generates a Hierarchical Clustering Dendrogram."""
    os.makedirs(output_dir, exist_ok=True)
    data = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    linked = linkage(data, 'ward')
    plt.figure(figsize=(12, 7))
    dendrogram(linked, labels=labels, orientation='top', distance_sort='descending')
    plt.title("Hierarchical Text Clustering")
    plt.xticks(rotation=90)
    plt.ylabel("Distance (Ward)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dendogram_{analysis}.png"))
    plt.close()