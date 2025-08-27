import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
import os
from tqdm import tqdm

class AdvancedEmojiTrainer:
    def __init__(self, model, train_loader, val_loader, config, model_name="unknown"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name  # Add model name for unique saving
        self.device = torch.device(config.DEVICE)
        
        # Move model to device
        self.model.to(self.device)
        
        # Use standard BCE loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Fixed scheduler
        # Simple learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            # self.optimizer, mode='min', factor=0.5, patience=3
            self.optimizer, step_size= 5, gamma=0.5
        )

        
        # Training tracking
        self.best_f1 = 0
        self.best_model_path = None  # Track the best model path
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        
        print(f"? {model_name} initialized with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        print(f"? Device: {self.device} | LR: {config.LEARNING_RATE} | Epochs: {config.NUM_EPOCHS}")
    
    def train_epoch(self, epoch):
        """Training epoch with progress bar"""
        self.model.train()
        total_loss = 0
        
        # Create progress bar
        pbar = tqdm(self.train_loader, desc=f"{self.model_name} Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validation with key metrics only"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        exact_match = np.mean(np.all(all_predictions == all_labels, axis=1))
        avg_f1 = self.calculate_f1(all_labels, all_predictions)
        
        val_loss = total_loss / len(self.val_loader)
        
        return {
            'loss': val_loss,
            'exact_match': exact_match,
            'f1': avg_f1
        }
    
    def calculate_f1(self, y_true, y_pred):
        """Calculate average F1 score"""
        f1_scores = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:
                f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                f1_scores.append(f1)
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def save_best_model(self, epoch, f1_score):
        """Save only the best model, delete previous if exists"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # New model path
        new_model_path = os.path.join(
            self.config.OUTPUT_DIR, 
            f"best_{self.model_name.lower()}_f1_{f1_score:.4f}.pth"
        )
        
        # Delete previous best model if exists
        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
            print(f"??  Deleted previous model: {os.path.basename(self.best_model_path)}")
        
        # Save new best model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'best_f1': f1_score,
            'epoch': epoch,
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores
        }, new_model_path)
        
        # Update best model path
        self.best_model_path = new_model_path
        print(f"? New best {self.model_name} saved: {os.path.basename(new_model_path)}")
    
    def train(self):
        """Clean training loop with smart model saving"""
        print(f"? Starting {self.model_name} training...")
        print("=" * 50)
        
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_f1_scores.append(val_metrics['f1'])
            
            # Update learning rate
            # self.scheduler.step(val_metrics['loss'])
            self.scheduler.step()

            
            # Print epoch summary
            print(f"? Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | Exact Match: {val_metrics['exact_match']:.4f}")
            
            # Save best model (replaces previous best)
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                patience_counter = 0
                self.save_best_model(epoch + 1, val_metrics['f1'])
                print(f"? New best F1: {self.best_f1:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"? Early stopping. Best F1: {self.best_f1:.4f}")
                break
        
        print(f"? {self.model_name} training completed! Best F1: {self.best_f1:.4f}")
        print(f"? Best model saved as: {os.path.basename(self.best_model_path) if self.best_model_path else 'None'}")
        
        return self.best_model_path  # Return path for ensemble use