import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Union, Optional
from transformers import AutoModel, AutoTokenizer, get_scheduler

# ==========================================
# EXAMPLE USAGE
# ==========================================
# if __name__ == "__main__":

#     # 1. Initialize with Hugging Face (e.g., for Text)
#     # pipeline = DeepPipeline("distilbert-base-uncased", task="clf")
#     # loader = ToolboxDataLoader.get_loader(..., tokenizer=pipeline.get_tokenizer())
#
#     # 2. Initialize with Custom Torch Model (e.g., for Images)
#     # my_model = MyCNN()
#     # pipeline = DeepPipeline(my_model, task="clf")
#     # pipeline.fit(train_loader, epochs=10)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DeepPipeline:
    """
    Unified Deep Learning Engine for the Toolbox.
    Supports Classification, Regression, and Segmentation.
    Integrates with Hugging Face and custom PyTorch models.
    """
    def __init__(self, model_name_or_obj: Union[nn.Module, str], task: str = "clf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.tokenizer = None
        
        # 1. Initialization: Hugging Face String vs. PyTorch Module
        if isinstance(model_name_or_obj, str):
            logger.info(f"Initializing Hugging Face assets: {model_name_or_obj}")
            self.model = AutoModel.from_pretrained(model_name_or_obj)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_obj)
            except Exception as e:
                logger.warning(f"No default tokenizer found for {model_name_or_obj}: {e}")
        else:
            self.model = model_name_or_obj
            
        self.model.to(self.device)
        logger.info(f"Model successfully loaded on {self.device}")

    def get_tokenizer(self) -> Optional[AutoTokenizer]:
        """Exposes the tokenizer to be passed into the ToolboxDataLoader."""
        return self.tokenizer

    def train_epoch(self, dataloader, optimizer, scheduler, criterion):
        """Standard training loop for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        for batch in dataloader:
            # Transfer all tensors in the tuple (x, mask, label) to device
            batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
            
            # Unpack based on our Dataset tuple structure: (Input, *Targets)
            inputs = batch[0]
            targets = batch[-1] # Ground truth is always the last element
            
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = self.model(inputs)
            
            # Extract logits if the model is a Transformers object
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            loss = criterion(logits, targets)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
        return running_loss / len(dataloader)

    def validate(self, dataloader, criterion):
        """Validation loop to monitor overfitting."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
                inputs, targets = batch[0], batch[-1]
                
                outputs = self.model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                loss = criterion(logits, targets)
                val_loss += loss.item()
                
        return val_loss / len(dataloader)

    def fit(self, train_loader, val_loader=None, epochs: int = 5, lr: float = 1e-4):
        """
        The main orchestrator for the training process.
        Automatically selects Loss and Scheduler.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # Learning Rate Scheduler (Linear decay)
        num_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_steps)
        
        # Automatic Criterion Selection based on task
        if self.task == "clf":
            criterion = nn.CrossEntropyLoss()
        elif self.task == "reg":
            criterion = nn.MSELoss()
        elif self.task == "seg":
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Task '{self.task}' is not supported by the DeepPipeline.")

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, lr_scheduler, criterion)
            
            status = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}"
            
            if val_loader:
                v_loss = self.validate(val_loader, criterion)
                status += f" | Val Loss: {v_loss:.4f}"
            
            logger.info(status)

    def save(self, path: str):
        """Save weights to disk."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load weights from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")