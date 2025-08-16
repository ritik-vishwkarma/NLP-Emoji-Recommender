import torch

class AdvancedConfig:
    # Dataset paths
    DATA_PATH = "massive_emoji_dataset.csv"  # Your expanded dataset
    
    # Model configurations
    MODELS_TO_TRAIN = {
        'mobilebert': 'google/mobilebert-uncased',
        'distilbert': 'distilbert-base-uncased',
        'roberta': 'roberta-base',
        'deepmoji': 'cardiffnlp/twitter-roberta-base-emoji',
        'xlnet': 'xlnet-base-cased',
        'multibert': 'bert-base-multilingual-cased'  # For Hindi support
    }
    
    # Training parameters - OPTIMIZED FOR LARGE DATASET
    BATCH_SIZE = 16  # Larger batch size for stable training
    LEARNING_RATE = 2e-5  # Optimal for transformers
    NUM_EPOCHS = 20  # More epochs for large dataset
    MAX_LENGTH = 128
    WARMUP_RATIO = 0.1  # 10% warmup
    WEIGHT_DECAY = 0.01
    
    # Advanced training features
    USE_FOCAL_LOSS = True  # Handle class imbalance
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING = 0.1
    USE_GRADIENT_ACCUMULATION = True
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # Data parameters
    MIN_EMOJI_FREQ = 10  # Higher frequency threshold
    MAX_SEQUENCE_LENGTH = 128
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Evaluation
    EARLY_STOPPING_PATIENCE = 5
    EVAL_STEPS = 500  # Evaluate every 500 steps
    SAVE_STEPS = 1000
    
    # Output
    OUTPUT_DIR = "models_advanced"
    RESULTS_DIR = "results_advanced"
    
    # Ensemble
    USE_ENSEMBLE = True
    ENSEMBLE_WEIGHTS = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]  # Model weights