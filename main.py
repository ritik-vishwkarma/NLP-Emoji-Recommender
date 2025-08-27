from nlp.src.models import MobileBERTEmojiClassifier, DistilBERTEmojiClassifier, RoBERTaEmojiClassifier, DeepMojiEmojiClassifier, XLNetEmojiClassifier, MultiHeadEmojiClassifier, EnsembleEmojiClassifier
from nlp.src.train import AdvancedEmojiTrainer
from nlp.src.config import AdvancedConfig
from nlp.src.data_processor import EnhancedEmojiDataProcessor, create_data_loaders
from transformers import AutoTokenizer
import torch

def test_model(model, test_loader, device):
    """Test model and return metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.4).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    import numpy as np
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    exact_match = np.mean(np.all(all_predictions == all_labels, axis=1))
    
    # Per-class F1 scores
    f1_scores = []
    for i in range(all_labels.shape[1]):
        if all_labels[:, i].sum() > 0:
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels[:, i], all_predictions[:, i], zero_division=0)
            f1_scores.append(f1)
    
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    return {
        'f1': avg_f1,
        'exact_match': exact_match,
        'num_classes_predicted': np.sum(all_predictions.sum(axis=0) > 0)
    }

def main_advanced():
    """Advanced training pipeline with multiple models"""
    config = AdvancedConfig()
    
    print("🚀 Advanced Emoji Prediction Training Pipeline")
    print("🌟 Multiple Models + Massive Dataset")
    print("=" * 60)
    
    # Step 1: Load and process the existing massive dataset
    print("\n📊 Loading massive dataset...")
    processor = EnhancedEmojiDataProcessor(config)
    df = processor.load_data('massive_emoji_dataset.csv')
    
    if df is None:
        print("❌ Error: Could not load massive_emoji_dataset.csv")
        print("💡 Make sure the file exists in the current directory")
        return None
    
    df, labels = processor.prepare_labels(df, min_freq=config.MIN_EMOJI_FREQ)
    data_splits, label_splits = processor.split_data(df, labels)
    
    print(f"✅ Dataset loaded: {len(df)} samples with {len(processor.emoji_to_id)} emoji classes")
    
    # Step 2: Train multiple models
    model_classes = {
        'MobileBERT': MobileBERTEmojiClassifier,
        'DistilBERT': DistilBERTEmojiClassifier,
        'RoBERTa': RoBERTaEmojiClassifier,
        'DeepMoji': DeepMojiEmojiClassifier,
        'XLNet': XLNetEmojiClassifier,
        'MultiHead': MultiHeadEmojiClassifier
        
    }
    
    trained_models = {}
    results = {}
    
    for model_name, model_class in model_classes.items():
        print(f"\n? Training {model_name}...")
    
        try:
            # Get appropriate tokenizer
            model_path = config.MODELS_TO_TRAIN.get(model_name.lower(), 'distilbert-base-uncased')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                data_splits, label_splits, tokenizer, config
            )
            
            # Initialize model
            num_classes = len(processor.emoji_to_id)
            if model_name == 'DeepMoji':
                model = model_class(num_classes=num_classes, model_name='cardiffnlp/twitter-roberta-base-emoji')
            else:
                model = model_class(num_classes=num_classes, model_name=model_path)
            
            # Train model with model name
            trainer = AdvancedEmojiTrainer(model, train_loader, val_loader, config, model_name)
            best_model_path = trainer.train()  # Get the best model path
            
            # Test and store results
            test_metrics = test_model(model, test_loader, config.DEVICE)
            results[model_name] = test_metrics
            results[model_name]['model_path'] = best_model_path  # Store model path
            trained_models[model_name] = model
            
            print(f"? {model_name} F1: {test_metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"? Error training {model_name}: {str(e)}")
            print(f"??  Skipping {model_name} and continuing with next model...")
            continue
    
    # Step 3: Create ensemble (if enabled and we have trained models)
    if config.USE_ENSEMBLE and len(trained_models) > 1:
        print("\n🎯 Creating ensemble model...")
        try:
            ensemble_models = list(trained_models.values())
            # Adjust weights to match number of trained models
            weights = config.ENSEMBLE_WEIGHTS[:len(ensemble_models)]
            if len(weights) < len(ensemble_models):
                weights.extend([1.0] * (len(ensemble_models) - len(weights)))
            
            ensemble = EnsembleEmojiClassifier(ensemble_models, weights)
            
            # Test ensemble
            ensemble_metrics = test_model(ensemble, test_loader, config.DEVICE)
            results['Ensemble'] = ensemble_metrics
            print(f"✅ Ensemble F1: {ensemble_metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"❌ Error creating ensemble: {str(e)}")
    
    # Step 4: Final comparison
    if results:
        print("\n📈 Final Results Comparison:")
        print("=" * 60)
        for model_name, metrics in results.items():
            print(f"{model_name:12} | F1: {metrics['f1']:.4f} | Exact: {metrics['exact_match']:.4f}")
        
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        print(f"\n🏆 Best Model: {best_model} (F1: {results[best_model]['f1']:.4f})")
    else:
        print("\n❌ No models were successfully trained!")
    
    return results

if __name__ == "__main__":
    results = main_advanced()