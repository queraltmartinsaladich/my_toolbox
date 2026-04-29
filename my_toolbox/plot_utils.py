import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
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

def plot_loss_curves(history: dict, save_path: str = "results/loss_curves.png"):
    """
    Plots training and validation loss/metrics over epochs.
    Expects history dict with lists: {'train_loss': [...], 'val_loss': [...]}
    """
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Optional: If you tracked accuracy
    if 'train_acc' in history:
        ax2 = plt.gca().twinx()
        ax2.plot(epochs, history['train_acc'], 'g--', label='Train Acc')
        if 'val_acc' in history:
            ax2.plot(epochs, history['val_acc'], 'y--', label='Val Acc')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Loss curves saved to {save_path}")

def plot_regression_results(y_true, y_pred, save_path: str):
    """
    Generates a scatter plot of Actual vs Predicted values with an identity line.
    Useful for seeing how far off your regression model is.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
    
    # Plot Identity Line (Perfect prediction line)
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit")
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression: Actual vs. Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_segmentation_sample(image, mask, prediction, save_path: str):
    """
    Visualizes the Input Image, Ground Truth Mask, and Model Prediction side-by-side.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Standardize image for plotting if it's a tensor (C, H, W) -> (H, W, C)
    if hasattr(image, 'permute'):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    ax[0].axis('off')
    
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis('off')
    
    ax[2].imshow(prediction, cmap='gray')
    ax[2].set_title("Model Prediction")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()