"""
DEEP LEARNING ORCHESTRATOR (main_deep.py)
-----------------------------------------
Purpose:
    Command-line interface for Modern Deep Learning workflows. 
    Handles high-dimensional data (Images) and Transformer-based NLP using PyTorch.

Key Capabilities:
    - Computer Vision: Support for Image Classification and Semantic Segmentation.
    - Transformer NLP: Integration with Hugging Face AutoModels and Tokenizers.
    - Hardware: Automatic CUDA/GPU detection and memory management.
    - Task Engines: Unified training loops for Classification, Regression, and Segmentation.

Input Requirements:
    - A Manifest CSV/JSON containing file paths (relative to ./data).
    - A valid model string (Hugging Face) or a local PyTorch model object.
    - Specified data type ('image' or 'text') to trigger specialized loading.

Usage Examples:
    Segmentation:   python main_deep.py --path manifest.csv --type image --task segmentation --model unet
    NLP Classify:   python main_deep.py --path data.csv --type text --task classification --model bert-base-uncased
    Image Classify: python main_deep.py --path data.csv --type image --task classification --model resnet50

Dependencies:
    torch, transformers, pandas, PIL/OpenCV
"""

import argparse
import logging
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Import your custom toolbox components
from my_toolbox import (
    ToolboxDataLoader, 
    DeepPipeline, 
    full_classification_metrics, 
    full_regression_metrics, 
    segmentation_metrics)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_deep_pipeline(args):
    try:
        os.makedirs("results", exist_ok=True)
        
        # 1. Load Manifest (The CSV containing paths/labels)
        df = pd.read_csv(args.path)
        if args.subset:
            df = df.sample(n=min(args.subset, len(df)), random_state=42)
            
        # 2. Split Data
        # Use stratification only for classification tasks
        strat = df[args.label] if args.task == 'clf' and args.label in df.columns else None
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=strat, random_state=42)

        # 3. Initialize Pipeline (Model + Device + Tokenizer)
        pipeline = DeepPipeline(model_name_or_obj=args.model, task=args.task)

        # 4. Initialize DataLoaders via Factory
        train_loader = ToolboxDataLoader.get_loader(
            data_type=args.type, 
            data=train_df, 
            batch_size=args.batch, 
            tokenizer=pipeline.get_tokenizer() # Auto-handles text vs image
        )
        
        val_loader = ToolboxDataLoader.get_loader(
            data_type=args.type, 
            data=val_df, 
            batch_size=args.batch,
            shuffle=False,
            tokenizer=pipeline.get_tokenizer()
        )

        # 5. Training Phase
        logger.info(f"Starting {args.task} training on {pipeline.device}...")
        pipeline.fit(
            train_loader=train_loader, 
            val_loader=val_loader, 
            epochs=args.epochs, 
            lr=args.lr
        )

        # 6. Evaluation Phase
        logger.info("Running final evaluation...")
        pipeline.model.eval()
        
        all_preds = []
        all_trues = []
        sample_img, sample_true, sample_pred = None, None, None

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # Unpack tuple: (Input, *Targets)
                batch = [item.to(pipeline.device) if isinstance(item, torch.Tensor) else item for item in batch]
                inputs, targets = batch[0], batch[-1]
                
                outputs = pipeline.model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Task-specific Prediction Logic
                if args.task == "classification":
                    preds = torch.argmax(logits, dim=1)
                elif args.task == "segmentation":
                    preds = (torch.sigmoid(logits) > 0.5).float()
                else: # Regression
                    preds = logits

                all_preds.append(preds.cpu())
                all_trues.append(targets.cpu())
                
                # Capture a sample for visual plotting (Segmentation)
                if i == 0:
                    sample_img, sample_true, sample_pred = inputs[0].cpu(), targets[0].cpu(), preds[0].cpu()

        # 7. Metrics Execution
        y_true = torch.cat(all_trues).numpy()
        y_pred = torch.cat(all_preds).numpy()

        if args.task == "clf":
            full_classification_metrics(y_true, y_pred, analysis="deep_run")
        elif args.task == "reg":
            full_regression_metrics(y_true, y_pred, analysis="deep_run")
        elif args.task == "seg":
            segmentation_metrics(sample_true, sample_pred, image=sample_img, analysis="deep_run")

        # 8. Save Model
        pipeline.save(f"results/{args.task}_model.pt")
        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Deep Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data Args
    parser.add_argument('--path', type=str, required=True, help="Path to manifest CSV")
    parser.add_argument('--type', type=str, choices=['image', 'text'], required=True)
    parser.add_argument('--task', type=str, choices=['clf', 'reg', 'seg'], required=True)
    
    # Model Args
    parser.add_argument('--model', type=str, required=True, help="Hugging Face string or local model path")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=16)
    
    # Mapping Args
    parser.add_argument('--label', type=str, default='label', help="Column name for target/label")
    parser.add_argument('--subset', type=int, default=None)
    
    args = parser.parse_args()
    run_deep_pipeline(args)