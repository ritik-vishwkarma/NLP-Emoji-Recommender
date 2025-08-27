import pandas as pd
import numpy as np
import ast
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import emoji
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from nlp.src.config import Config

class EnhancedEmojiDataProcessor:
    def __init__(self, config=None):
        self.config = config or Config()
        self.mlb = MultiLabelBinarizer()
        self.emoji_to_id = {}
        self.id_to_emoji = {}
        
    def load_data(self, file_path):
        """Load and preprocess the enhanced emoji dataset"""
        print(f"📂 Loading enhanced dataset from {file_path}")
        
        # Check if file exists
        import os
        if not os.path.exists(file_path):
            print(f"❌ ERROR: File {file_path} not found!")
            return None
        
        print(f"📊 File found, size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        try:
            # Load with progress indication
            print("🔄 Reading enhanced CSV file...")
            df = pd.read_csv(file_path)
            print(f"✅ CSV loaded successfully! Shape: {df.shape}")
            
            # Show dataset composition
            self._analyze_dataset_composition(df)
            
            # Clean the data
            print("🧹 Cleaning data...")
            df = self._clean_data(df)
            print(f"✅ After cleaning: {df.shape}")
            
            # Extract emojis
            print("😊 Extracting emojis...")
            df['emoji_list'] = df['emojis'].apply(self._parse_emoji_list)
            print("✅ Emoji extraction completed")
            
            # Filter out rows with no valid emojis
            print("🔍 Filtering valid emojis...")
            df = df[df['emoji_list'].apply(len) > 0]
            
            print(f"✅ Final dataset: {len(df)} samples with valid emojis")
            return df
            
        except Exception as e:
            print(f"❌ ERROR loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
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
        
        # Sample examples from each category
        print("\n📝 Sample Examples:")
        for source in df['dataset_source'].unique()[:3]:
            sample = df[df['dataset_source'] == source]['text'].iloc[0]
            print(f"  {source}: '{sample[:80]}...'")
    
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
        
        # Optional: Filter to only augmented data for testing
        if hasattr(self.config, 'FILTER_AUGMENTED_ONLY') and self.config.FILTER_AUGMENTED_ONLY:
            print("🎯 Filtering to augmented data only...")
            df = df[df['dataset_source'] == 'augmented']
            print(f"After augmented filter: {df.shape}")
        
        return df
    
    def _clean_text(self, text):
        """Clean individual text samples"""
        try:
            # Convert to string if not already
            text = str(text)
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Keep original text mostly intact for multilingual support
            # Only remove extreme special characters
            text = re.sub(r'[^\w\s\.,!?;:\-\'\"#@।]', '', text)
            
            return text
        except:
            return ""
    
    def _parse_emoji_list(self, emoji_str):
        """Parse emoji string to list of emojis - enhanced for your dataset"""
        try:
            # Handle string representation of list
            if isinstance(emoji_str, str):
                emoji_list = ast.literal_eval(emoji_str)
            else:
                emoji_list = emoji_str
            
            # Ensure it's a list
            if not isinstance(emoji_list, list):
                return []
            
            # Filter valid emojis and clean unicode issues
            valid_emojis = []
            for e in emoji_list:
                if isinstance(e, str) and len(e) > 0:
                    # Handle unicode issues in your dataset
                    cleaned_emoji = e.replace('\u200d', '').replace('️', '').strip()
                    if len(cleaned_emoji) > 0:
                        valid_emojis.append(cleaned_emoji)
            
            return valid_emojis
        except Exception as e:
            # If parsing fails, return empty list
            return []
    
    def prepare_labels(self, df, min_freq=None):
        """Prepare emoji labels for training - enhanced version"""
        min_freq = min_freq or getattr(self.config, 'MIN_EMOJI_FREQ', 5)
        print(f"🏷️ Preparing labels with min_freq={min_freq}")
        
        # Count emoji frequencies
        all_emojis = []
        for emoji_list in df['emoji_list']:
            all_emojis.extend(emoji_list)
        
        emoji_counts = Counter(all_emojis)
        print(f"📊 Total unique emojis: {len(emoji_counts)}")
        
        # Show top 15 most frequent emojis
        print("🔝 Top 15 most frequent emojis:")
        for emoji, count in emoji_counts.most_common(15):
            print(f"  {emoji}: {count}")
        
        # Filter emojis by minimum frequency
        frequent_emojis = [emoji for emoji, count in emoji_counts.items() if count >= min_freq]
        
        print(f"✅ Using {len(frequent_emojis)} emojis with frequency >= {min_freq}")
        
        if len(frequent_emojis) == 0:
            print("❌ No emojis meet the minimum frequency requirement!")
            print("💡 Try reducing min_freq parameter")
            return df, None
        
        # Create mappings
        self.emoji_to_id = {emoji: i for i, emoji in enumerate(frequent_emojis)}
        self.id_to_emoji = {i: emoji for emoji, i in self.emoji_to_id.items()}
        
        # Filter dataframe to only include frequent emojis
        df['filtered_emojis'] = df['emoji_list'].apply(
            lambda x: [e for e in x if e in self.emoji_to_id]
        )
        
        # Remove samples with no frequent emojis
        df = df[df['filtered_emojis'].apply(len) > 0]
        print(f"📋 Final samples after filtering: {len(df)}")
        
        # Show data distribution by source after filtering
        if 'dataset_source' in df.columns:
            print("📊 Final data distribution by source:")
            for source in df['dataset_source'].value_counts().index:
                count = len(df[df['dataset_source'] == source])
                percentage = (count / len(df)) * 100
                print(f"  {source}: {count} ({percentage:.1f}%)")
        
        # Create binary labels
        print("🔢 Creating binary labels...")
        emoji_labels = self.mlb.fit_transform(df['filtered_emojis'])
        print(f"✅ Label matrix shape: {emoji_labels.shape}")
        
        return df, emoji_labels
    
    def split_data(self, df, labels):
        """Split data into train/val/test sets with robust stratification"""
        print("🔄 Splitting data with robust stratification...")
        
        # Create a simplified stratification key for sources with enough samples
        if 'dataset_source' in df.columns:
            source_counts = df['dataset_source'].value_counts()
            print(f"📊 Source distribution before split:")
            for source, count in source_counts.items():
                print(f"  {source}: {count}")
            
            # Group small sources together for stratification
            min_samples_for_stratify = 10  # Need at least 10 samples per group
            df['stratify_key'] = df['dataset_source'].apply(
                lambda x: x if source_counts[x] >= min_samples_for_stratify else 'other'
            )
            
            stratify_counts = df['stratify_key'].value_counts()
            print(f"\n📋 Stratification groups:")
            for key, count in stratify_counts.items():
                print(f"  {key}: {count}")
            
            # Use stratify_key if we have enough diversity, otherwise None
            stratify_col = df['stratify_key'] if len(stratify_counts) > 1 and min(stratify_counts) >= 2 else None
        else:
            stratify_col = None
        
        try:
            # First split: train + val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                df, labels, 
                test_size=self.config.TEST_SPLIT, 
                random_state=self.config.RANDOM_SEED, 
                stratify=stratify_col
            )
            
            # Update stratify column for second split
            if stratify_col is not None:
                temp_stratify = X_temp['stratify_key']
                temp_counts = temp_stratify.value_counts()
                temp_stratify = temp_stratify if len(temp_counts) > 1 and min(temp_counts) >= 2 else None
            else:
                temp_stratify = None
            
            # Second split: train vs val
            val_size = self.config.VAL_SPLIT / (self.config.TRAIN_SPLIT + self.config.VAL_SPLIT)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_size, 
                random_state=self.config.RANDOM_SEED,
                stratify=temp_stratify
            )
            
        except ValueError as e:
            print(f"⚠️  Stratification failed: {e}")
            print("🔄 Falling back to random split without stratification...")
            
            # Fallback to random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                df, labels, 
                test_size=self.config.TEST_SPLIT, 
                random_state=self.config.RANDOM_SEED, 
                stratify=None
            )
            
            val_size = self.config.VAL_SPLIT / (self.config.TRAIN_SPLIT + self.config.VAL_SPLIT)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_size, 
                random_state=self.config.RANDOM_SEED,
                stratify=None
            )
        
        print(f"📊 Data split completed:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples") 
        print(f"  Test: {len(X_test)} samples")
        
        # Show split composition
        if 'dataset_source' in df.columns:
            for split_name, split_data in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
                print(f"\n{split_name} split composition:")
                source_dist = split_data['dataset_source'].value_counts()
                for source, count in source_dist.items():
                    percentage = (count / len(split_data)) * 100
                    print(f"    {source}: {count} ({percentage:.1f}%)")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)

class EmojiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]
        
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

def create_data_loaders(data_splits, label_splits, tokenizer, config):
    """Create data loaders for training"""
    X_train, X_val, X_test = data_splits
    y_train, y_val, y_test = label_splits
    
    # Create datasets
    train_dataset = EmojiDataset(X_train['text'], y_train, tokenizer, config.MAX_LENGTH)
    val_dataset = EmojiDataset(X_val['text'], y_val, tokenizer, config.MAX_LENGTH)
    test_dataset = EmojiDataset(X_test['text'], y_test, tokenizer, config.MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Add the missing test function and main execution
def test_enhanced_data():
    """Test the enhanced emoji dataset loading and processing"""
    config = Config()
    
    print("🧪 Testing Enhanced Emoji Dataset Loading")
    print("=" * 50)
    
    # Check if enhanced file exists
    import os
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ Enhanced dataset not found: {config.DATA_PATH}")
        return False
    
    # Initialize processor
    processor = EnhancedEmojiDataProcessor(config)
    
    # Load data
    df = processor.load_data(config.DATA_PATH)
    
    if df is not None:
        print(f"\n✅ Successfully loaded {len(df)} samples")
        
        # Test label preparation
        df, labels = processor.prepare_labels(df, min_freq=3)
        
        if labels is not None:
            print(f"✅ Successfully prepared {labels.shape[1]} emoji classes")
            
            # Test data splitting
            data_splits, label_splits = processor.split_data(df, labels)
            print(f"✅ Successfully split data")
            
            # Test data loader creation (optional)
            print("\n🔧 Testing data loader creation...")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
                train_loader, val_loader, test_loader = create_data_loaders(
                    data_splits, label_splits, tokenizer, config
                )
                print(f"✅ Data loaders created successfully")
                print(f"   Train batches: {len(train_loader)}")
                print(f"   Val batches: {len(val_loader)}")
                print(f"   Test batches: {len(test_loader)}")
                
                # Test one batch
                sample_batch = next(iter(train_loader))
                print(f"✅ Sample batch shape:")
                print(f"   Input IDs: {sample_batch['input_ids'].shape}")
                print(f"   Attention Mask: {sample_batch['attention_mask'].shape}")
                print(f"   Labels: {sample_batch['labels'].shape}")
                
            except Exception as e:
                print(f"⚠️  Data loader test failed: {e}")
            
            return True
    
    return False

if __name__ == "__main__":
    success = test_enhanced_data()
    if success:
        print("\n🎉 Enhanced dataset is ready for training!")
        print("📝 Next steps:")
        print("   1. Run: python main.py")
        print("   2. Or train specific models with models.py and trainer.py")
    else:
        print("\n❌ Issues found with enhanced dataset")
        print("💡 Check your dataset file and config.py")