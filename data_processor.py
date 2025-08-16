import pandas as pd
import numpy as np
import ast
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from config import AdvancedConfig

class EnhancedEmojiDataProcessor:
    def __init__(self, config=None):
        self.config = config or AdvancedConfig()
        self.mlb = MultiLabelBinarizer()
        self.emoji_to_id = {}
        self.id_to_emoji = {}
        self.class_weights = None
        
    def load_data(self, file_path):
        """Load and preprocess the enhanced emoji dataset"""
        print(f"📂 Loading enhanced dataset from {file_path}")
        
        import os
        if not os.path.exists(file_path):
            print(f"❌ ERROR: File {file_path} not found!")
            return None
        
        print(f"📊 File found, size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ CSV loaded successfully! Shape: {df.shape}")
            
            self._analyze_dataset_composition(df)
            df = self._clean_data(df)
            print(f"✅ After cleaning: {df.shape}")
            
            # Extract emojis with better handling
            print("😊 Extracting emojis...")
            df['emoji_list'] = df['emojis'].apply(self._parse_emoji_list)
            print("✅ Emoji extraction completed")
            
            # Filter valid emojis
            df = df[df['emoji_list'].apply(len) > 0]
            print(f"✅ Final dataset: {len(df)} samples with valid emojis")
            
            return df
            
        except Exception as e:
            print(f"❌ ERROR loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_emoji_list(self, emoji_str):
        """Enhanced emoji parsing with better unicode handling"""
        try:
            if isinstance(emoji_str, str):
                emoji_list = ast.literal_eval(emoji_str)
            else:
                emoji_list = emoji_str
            
            if not isinstance(emoji_list, list):
                return []
            
            valid_emojis = []
            for e in emoji_list:
                if isinstance(e, str) and len(e) > 0:
                    # Better unicode cleaning
                    cleaned_emoji = e.replace('\u200d', '').replace('️', '').strip()
                    
                    # Remove duplicates like 😂😂 -> 😂
                    if len(cleaned_emoji) > 2 and cleaned_emoji[0] == cleaned_emoji[1]:
                        cleaned_emoji = cleaned_emoji[0]
                    
                    if len(cleaned_emoji) > 0 and cleaned_emoji not in valid_emojis:
                        valid_emojis.append(cleaned_emoji)
            
            return valid_emojis[:3]  # Limit to top 3 emojis per sample
        except:
            return []
    
    def prepare_labels(self, df, min_freq=None):
        """Enhanced label preparation with class balancing"""
        min_freq = min_freq or getattr(self.config, 'MIN_EMOJI_FREQ', 2)
        print(f"🏷️ Preparing labels with min_freq={min_freq}")
        
        # Count emoji frequencies
        all_emojis = []
        for emoji_list in df['emoji_list']:
            all_emojis.extend(emoji_list)
        
        emoji_counts = Counter(all_emojis)
        print(f"📊 Total unique emojis: {len(emoji_counts)}")
        
        # Show top emojis
        print("🔝 Top 20 most frequent emojis:")
        for emoji, count in emoji_counts.most_common(20):
            print(f"  {emoji}: {count}")
        
        # More inclusive emoji selection
        frequent_emojis = [emoji for emoji, count in emoji_counts.items() if count >= min_freq]
        
        print(f"✅ Using {len(frequent_emojis)} emojis with frequency >= {min_freq}")
        
        if len(frequent_emojis) < 10:
            print("⚠️  Very few emojis found, reducing min_freq to 1")
            frequent_emojis = [emoji for emoji, count in emoji_counts.items() if count >= 1]
            print(f"✅ Now using {len(frequent_emojis)} emojis")
        
        # Create mappings
        self.emoji_to_id = {emoji: i for i, emoji in enumerate(frequent_emojis)}
        self.id_to_emoji = {i: emoji for emoji, i in self.emoji_to_id.items()}
        
        # Filter dataframe
        df['filtered_emojis'] = df['emoji_list'].apply(
            lambda x: [e for e in x if e in self.emoji_to_id]
        )
        
        df = df[df['filtered_emojis'].apply(len) > 0]
        print(f"📋 Final samples after filtering: {len(df)}")
        
        # Calculate class weights for imbalanced data
        emoji_frequencies = np.zeros(len(frequent_emojis))
        for emoji_list in df['filtered_emojis']:
            for emoji in emoji_list:
                emoji_frequencies[self.emoji_to_id[emoji]] += 1
        
        # Compute inverse frequency weights
        total_samples = len(df)
        self.class_weights = total_samples / (len(frequent_emojis) * emoji_frequencies + 1e-6)
        self.class_weights = torch.FloatTensor(self.class_weights)
        
        print(f"📊 Class weights computed for {len(frequent_emojis)} classes")
        
        # Create binary labels
        emoji_labels = self.mlb.fit_transform(df['filtered_emojis'])
        print(f"✅ Label matrix shape: {emoji_labels.shape}")
        
        return df, emoji_labels
    
    # ... (keep rest of the methods the same as before)
    def _analyze_dataset_composition(self, df):
        """Analyze the enhanced dataset composition"""
        print("\n📈 Enhanced Dataset Analysis:")
        print("-" * 50)
        
        # Dataset source breakdown
        print("📋 Dataset Sources:")
        source_counts = df['dataset_source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
        
        # Language composition
        print("\n🌍 Language Composition:")
        print(f"  Hindi content: {df['has_hindi'].sum()} ({(df['has_hindi'].sum()/len(df))*100:.1f}%)")
        print(f"  English content: {df['has_english'].sum()} ({(df['has_english'].sum()/len(df))*100:.1f}%)")
        print(f"  Code-mixed: {df['is_code_mixed'].sum()} ({(df['is_code_mixed'].sum()/len(df))*100:.1f}%)")
        
        # Show augmented vs original data
        augmented_count = len(df[df['dataset_source'] == 'augmented'])
        original_count = len(df[df['dataset_source'] != 'augmented'])
        print(f"\n🔧 Data Composition:")
        print(f"  Original data: {original_count} ({(original_count/len(df))*100:.1f}%)")
        print(f"  Augmented data: {augmented_count} ({(augmented_count/len(df))*100:.1f}%)")
    
    def _clean_data(self, df):
        """Clean and preprocess the enhanced dataset"""
        print(f"🔧 Initial shape: {df.shape}")
        
        # Check for required columns
        required_cols = ['text', 'emojis']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return df
        
        # Remove rows with missing text or emojis
        print("🗑️ Removing rows with missing data...")
        df = df.dropna(subset=['text', 'emojis'])
        print(f"After removing NaN: {df.shape}")
        
        # Clean text
        print("🧽 Cleaning text...")
        df['text'] = df['text'].apply(self._clean_text)
        
        # Filter by text length (minimum 3 characters)
        df = df[df['text'].str.len() >= 3]
        print(f"After text filtering: {df.shape}")
        
        return df
    
    def _clean_text(self, text):
        """Clean individual text samples"""
        try:
            text = str(text)
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Keep original text mostly intact for multilingual support
            text = re.sub(r'[^\w\s\.,!?;:\-\'\"#@।]', '', text)
            return text
        except:
            return ""
    
    def split_data(self, df, labels):
        """Enhanced data splitting"""
        print("🔄 Splitting data with robust stratification...")
        
        # Simple random split for multilabel data
        test_size = getattr(self.config, 'TEST_SPLIT', 0.1)
        X_temp, X_test, y_temp, y_test = train_test_split(
            df, labels, 
            test_size=test_size, 
            random_state=getattr(self.config, 'RANDOM_SEED', 42)
        )
        
        val_ratio = getattr(self.config, 'VAL_SPLIT', 0.1)
        train_ratio = getattr(self.config, 'TRAIN_SPLIT', 0.8)
        val_size_from_temp = val_ratio / (train_ratio + val_ratio)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_from_temp, 
            random_state=getattr(self.config, 'RANDOM_SEED', 42)
        )
        
        print(f"\n📊 Final data split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Val: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)

# Enhanced Dataset with data augmentation support
class EmojiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False):
        if hasattr(texts, 'reset_index'):
            texts = texts.reset_index(drop=True)
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if hasattr(self.texts, 'iloc'):
            text = str(self.texts.iloc[idx])
        else:
            text = str(self.texts[idx])
            
        label = self.labels[idx]
        
        # Simple data augmentation for training
        if self.augment and np.random.random() < 0.3:
            # Random lowercase/uppercase changes
            if np.random.random() < 0.5:
                text = text.lower()
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def create_data_loaders(data_splits, label_splits, tokenizer, config, class_weights=None):
    """Create enhanced data loaders"""
    X_train, X_val, X_test = data_splits
    y_train, y_val, y_test = label_splits
    
    # Create datasets with augmentation for training
    train_dataset = EmojiDataset(X_train['text'], y_train, tokenizer, 
                                getattr(config, 'MAX_LENGTH', 128), augment=True)
    val_dataset = EmojiDataset(X_val['text'], y_val, tokenizer, 
                              getattr(config, 'MAX_LENGTH', 128), augment=False)
    test_dataset = EmojiDataset(X_test['text'], y_test, tokenizer, 
                               getattr(config, 'MAX_LENGTH', 128), augment=False)
    
    batch_size = getattr(config, 'BATCH_SIZE', 8)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader