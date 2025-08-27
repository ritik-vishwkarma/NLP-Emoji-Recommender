import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np

class MobileBERTEmojiClassifier(nn.Module):
    """MobileBERT-based emoji classifier - lightweight and efficient"""
    def __init__(self, num_classes, model_name="google/mobilebert-uncased", dropout=0.3):
        super().__init__()
        self.mobilebert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.mobilebert.config.hidden_size, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.mobilebert(input_ids=input_ids, attention_mask=attention_mask)
        # MobileBERT uses pooler output
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

class DistilBERTEmojiClassifier(nn.Module):
    """Lighter but effective BERT variant"""
    def __init__(self, num_classes, model_name="distilbert-base-uncased", dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

class RoBERTaEmojiClassifier(nn.Module):
    """RoBERTa-based emoji classifier"""
    def __init__(self, num_classes, model_name="roberta-base", dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

class DeepMojiEmojiClassifier(nn.Module):
    """DeepMoji-style classifier using pre-trained emoji model"""
    def __init__(self, num_classes, model_name="cardiffnlp/twitter-roberta-base-emoji", dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-layer classifier for better emoji understanding
        hidden_size = self.roberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

class XLNetEmojiClassifier(nn.Module):
    """XLNet-based emoji classifier"""
    def __init__(self, num_classes, model_name="xlnet-base-cased", dropout=0.3):
        super().__init__()
        self.xlnet = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlnet.config.hidden_size, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)  # Mean pooling for XLNet
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

class MultiHeadEmojiClassifier(nn.Module):
    """Multi-head attention classifier"""
    def __init__(self, num_classes, model_name="distilbert-base-uncased", dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Multiple classification heads
        self.sentiment_head = nn.Linear(hidden_size, 8)  # 8 basic sentiments
        self.emotion_head = nn.Linear(hidden_size, 16)   # 16 emotions
        self.emoji_head = nn.Linear(hidden_size + 8 + 16, num_classes)  # Final emoji prediction
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.sentiment_head, self.emotion_head, self.emoji_head]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(
            sequence_output.transpose(0, 1),
            sequence_output.transpose(0, 1),
            sequence_output.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        
        # Pool the attended output
        pooled = attn_output.mean(dim=1)
        pooled = self.dropout(pooled)
        
        # Multi-stage prediction
        sentiment_logits = self.sentiment_head(pooled)
        emotion_logits = self.emotion_head(pooled)
        
        # Combine features
        combined_features = torch.cat([pooled, sentiment_logits, emotion_logits], dim=-1)
        emoji_logits = self.emoji_head(combined_features)
        
        return emoji_logits

class EnsembleEmojiClassifier(nn.Module):
    """Ensemble of multiple models"""
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0] * len(models)
        
    def forward(self, input_ids, attention_mask):
        outputs = []
        for i, model in enumerate(self.models):
            output = model(input_ids, attention_mask)
            outputs.append(output * self.weights[i])
        
        # Weighted average
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output