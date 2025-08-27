import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models from the correct location (root directory, not src)
try:
    from models import (
        MobileBERTEmojiClassifier, 
        DeepMojiEmojiClassifier, 
        DistilBERTEmojiClassifier,
        MultiHeadEmojiClassifier
    )
    from config import AdvancedConfig as Config
    logger.info("✅ Successfully imported models from root directory")
except ImportError as e:
    logger.error(f"❌ Could not import models: {e}")
    logger.error("Please ensure models.py and config.py exist in the same directory as recommender.py")
    raise

# Pydantic models for API
class TextInput(BaseModel):
    text: str
    top_k: int = 5
    model_type: str = "deepmoji"  # Default to best performing model

class BatchTextInput(BaseModel):
    texts: List[str]
    top_k: int = 5
    model_type: str = "deepmoji"

class EmojiPrediction(BaseModel):
    emoji: str
    label: str
    confidence: float
    rank: int

class PredictionResponse(BaseModel):
    text: str
    predictions: List[EmojiPrediction]
    model_used: str
    processing_time: float
    f1_score: float

class EmojiPredictor:
    def __init__(self, model, tokenizer, emoji_to_id, id_to_emoji, device, f1_score):
        self.model = model
        self.tokenizer = tokenizer
        self.emoji_to_id = emoji_to_id
        self.id_to_emoji = id_to_emoji
        self.device = device
        self.f1_score = f1_score
        self.model.eval()
    
    def predict(self, text: str, top_k: int = 5) -> tuple:
        """Predict emojis for a single text"""
        start_time = time.time()
        
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Get predictions
            outputs = self.model(**inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Apply sigmoid for multi-label classification
            probabilities = torch.sigmoid(logits)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.id_to_emoji)))
            
            results = []
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                emoji_id = idx.item()
                emoji = self.id_to_emoji.get(emoji_id, "❓")
                confidence = prob.item() * 100
                
                results.append({
                    'emoji': emoji,
                    'label': f'emoji_{emoji_id}',
                    'confidence': confidence,
                    'rank': i + 1
                })
        
        processing_time = time.time() - start_time
        return results, processing_time

class EmojiRecommendationSystem:
    def __init__(self):
        # Initialize config
        try:
            self.config = Config()
            logger.info("✅ Config loaded successfully")
        except Exception as e:
            logger.error(f"❌ Could not load config: {e}")
            # Fallback config
            class FallbackConfig:
                DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                OUTPUT_DIR = "models_advanced"
            self.config = FallbackConfig()
            logger.warning("⚠️ Using fallback config")
            
        self.models = {}
        self.tokenizers = {}
        self.predictors = {}
        
        # Load emoji mappings
        self._load_emoji_mappings()
        
        # Initialize only the 3 specified models
        self._load_specified_models()
    
    def _load_emoji_mappings(self):
        """Load emoji mappings from various possible locations"""
        mapping_paths = [
            Path("models_advanced/emoji_mappings.json"),
            Path("models/emoji_mappings.json"),
            Path("emoji_mappings.json"),
            Path("../models/emoji_mappings.json"),
            Path("data/emoji_mappings.json")
        ]
        
        mapping_loaded = False
        for mapping_path in mapping_paths:
            if mapping_path.exists():
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        mappings = json.load(f)
                        self.emoji_to_id = mappings['emoji_to_id']
                        self.id_to_emoji = {int(k): v for k, v in mappings['id_to_emoji'].items()}
                    logger.info(f"✅ Loaded emoji mappings from: {mapping_path}")
                    mapping_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load mappings from {mapping_path}: {e}")
        
        if not mapping_loaded:
            logger.error("❌ Emoji mappings file not found in any expected location!")
            logger.error("Please ensure emoji_mappings.json exists in one of these locations:")
            for path in mapping_paths:
                logger.error(f"  - {path}")
            raise FileNotFoundError("emoji_mappings.json not found")
        
        self.num_classes = len(self.emoji_to_id)
        logger.info(f"📊 Loaded {self.num_classes} emoji classes")
    
    def _load_specified_models(self):
        """Load only the 3 specified models from models_advanced directory"""
        
        # Define the exact 3 models we want to load
        model_configs = {
            'deepmoji': {
                'class': DeepMojiEmojiClassifier,
                'tokenizer': 'cardiffnlp/twitter-roberta-base-emoji',  # Correct tokenizer for DeepMoji
                'checkpoint_pattern': 'best_deepmoji_f1_*.pth'
            },
            'distilbert': {
                'class': DistilBERTEmojiClassifier,
                'tokenizer': 'distilbert-base-uncased',
                'checkpoint_pattern': 'best_distilbert_f1_*.pth'
            },
            'mobilebert': {
                'class': MobileBERTEmojiClassifier,
                'tokenizer': 'google/mobilebert-uncased',
                'checkpoint_pattern': 'best_mobilebert_f1_*.pth'
            }
        }
        
        # Check for models_advanced directory
        model_dirs = [
            Path("models_advanced"),
            Path("../models_advanced"),
            Path("./models_advanced")
        ]
        
        models_dir = None
        for model_dir in model_dirs:
            if model_dir.exists() and model_dir.is_dir():
                models_dir = model_dir
                logger.info(f"📁 Found models directory: {model_dir}")
                break
        
        if models_dir is None:
            logger.error("❌ models_advanced directory not found!")
            logger.error("Please ensure models_advanced directory exists with trained models")
            raise FileNotFoundError("models_advanced directory not found")
        
        # Load each specified model
        models_loaded = 0
        for model_name, config in model_configs.items():
            logger.info(f"🔍 Looking for {model_name} model...")
            
            # Find model file using pattern
            model_files = list(models_dir.glob(config['checkpoint_pattern']))
            
            if not model_files:
                logger.warning(f"❌ {model_name.upper()} model not found!")
                logger.warning(f"   Looking for pattern: {config['checkpoint_pattern']}")
                logger.warning(f"   In directory: {models_dir}")
                continue
            
            # Use the first matching file (or could sort by F1 score if multiple)
            checkpoint_path = model_files[0]
            logger.info(f"✅ Found {model_name} model: {checkpoint_path.name}")
            
            try:
                # Load tokenizer
                logger.info(f"📥 Loading tokenizer: {config['tokenizer']}")
                tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
                
                # Initialize model
                logger.info(f"🤖 Initializing {model_name} model with {self.num_classes} classes")
                model = config['class'](num_classes=self.num_classes)
                
                # Load checkpoint
                logger.info(f"📦 Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
                
                # Extract F1 score from filename or checkpoint
                f1_score = 0.0
                if 'best_f1' in checkpoint:
                    f1_score = checkpoint['best_f1']
                else:
                    # Try to extract from filename
                    import re
                    f1_match = re.search(r'f1_(\d+\.\d+)', checkpoint_path.name)
                    if f1_match:
                        f1_score = float(f1_match.group(1))
                
                # Load model state
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("✅ Loaded model_state_dict from checkpoint")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    logger.info("✅ Loaded state_dict from checkpoint")
                else:
                    model.load_state_dict(checkpoint)
                    logger.info("✅ Loaded checkpoint directly")
                
                model.to(self.config.DEVICE)
                model.eval()
                
                # Store components
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.predictors[model_name] = EmojiPredictor(
                    model, tokenizer, self.emoji_to_id, self.id_to_emoji, 
                    self.config.DEVICE, f1_score
                )
                
                models_loaded += 1
                logger.info(f"🎉 Successfully loaded {model_name.upper()} (F1: {f1_score:.4f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name} model: {e}")
                import traceback
                traceback.print_exc()
        
        if models_loaded == 0:
            logger.error("❌ No models could be loaded!")
            logger.error("Please ensure the following model files exist in models_advanced/:")
            for model_name, config in model_configs.items():
                logger.error(f"  - {config['checkpoint_pattern']}")
            raise RuntimeError("No models could be loaded!")
        
        logger.info(f"🎉 Successfully loaded {models_loaded}/3 models: {list(self.models.keys())}")
        
        # Display model performance ranking
        if self.models:
            model_performance = [(name, self.predictors[name].f1_score) for name in self.models.keys()]
            model_performance.sort(key=lambda x: x[1], reverse=True)
            logger.info("📊 Model Performance Ranking:")
            for i, (name, f1) in enumerate(model_performance, 1):
                logger.info(f"  {i}. {name.upper()}: F1 = {f1:.4f}")
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models with their F1 scores"""
        models_info = []
        for name in self.models.keys():
            models_info.append({
                'name': name,
                'f1_score': self.predictors[name].f1_score,
                'display_name': name.upper()
            })
        
        # Sort by F1 score (best first)
        models_info.sort(key=lambda x: x['f1_score'], reverse=True)
        return models_info
    
    def recommend_emojis(self, text: str, top_k: int = 5, model_type: str = "deepmoji"):
        """Recommend emojis for given text using specified model"""
        if model_type not in self.predictors:
            available = list(self.predictors.keys())
            if available:
                # Use the best performing model as fallback
                best_model = max(available, key=lambda x: self.predictors[x].f1_score)
                logger.warning(f"⚠️ Model '{model_type}' not available, using best model: {best_model}")
                model_type = best_model
            else:
                raise HTTPException(status_code=500, detail="No models available")
        
        predictions, processing_time = self.predictors[model_type].predict(text, top_k)
        
        return {
            'text': text,
            'predictions': predictions,
            'model_used': model_type,
            'processing_time': processing_time,
            'f1_score': self.predictors[model_type].f1_score
        }
    
    def batch_recommend(self, texts: List[str], top_k: int = 5, model_type: str = "deepmoji"):
        """Recommend emojis for multiple texts"""
        if model_type not in self.predictors:
            available = list(self.predictors.keys())
            if available:
                best_model = max(available, key=lambda x: self.predictors[x].f1_score)
                model_type = best_model
            else:
                raise HTTPException(status_code=500, detail="No models available")
        
        results = []
        for text in texts:
            result = self.recommend_emojis(text, top_k, model_type)
            results.append(result)
        
        return results

# Initialize the recommendation system
try:
    logger.info("🚀 Initializing Emoji Recommendation System...")
    recommendation_system = EmojiRecommendationSystem()
    logger.info("✅ Recommendation system initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize recommendation system: {e}")
    import traceback
    traceback.print_exc()
    recommendation_system = None

# ...existing code... (FastAPI app and routes remain the same)

# FastAPI app
app = FastAPI(
    title="🤖 Emoji Recommendation API",
    description="AI-powered emoji recommendation system with DeepMoji, DistilBERT, and MobileBERT models",
    version="2.0.0"
)

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 Emoji Recommender - Advanced Models</title>
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                margin-bottom: 30px;
                opacity: 0.8;
                font-size: 1.1em;
            }
            .input-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
            }
            input, select, textarea {
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                background: rgba(255, 255, 255, 0.9);
                color: #333;
                box-sizing: border-box;
            }
            textarea {
                height: 100px;
                resize: vertical;
            }
            button {
                background: #ff6b6b;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 10px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
            }
            button:hover {
                background: #ff5252;
                transform: translateY(-2px);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .results {
                margin-top: 30px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                display: none;
            }
            .prediction {
                display: flex;
                align-items: center;
                margin: 10px 0;
                padding: 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                transition: transform 0.2s ease;
            }
            .prediction:hover {
                transform: translateX(5px);
                background: rgba(255, 255, 255, 0.15);
            }
            .emoji {
                font-size: 2.5em;
                margin-right: 20px;
            }
            .prediction-info {
                flex: 1;
            }
            .rank {
                font-weight: bold;
                font-size: 1.1em;
            }
            .confidence {
                margin-left: auto;
                font-weight: bold;
                font-size: 1.2em;
                color: #4CAF50;
            }
            .loading {
                text-align: center;
                font-style: italic;
                padding: 20px;
            }
            .model-info {
                text-align: center;
                margin-top: 15px;
                padding: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                font-size: 0.9em;
            }
            .f1-score {
                color: #4CAF50;
                font-weight: bold;
            }
            .flex-row {
                display: grid;
                grid-template-columns: 2fr 1fr 1fr;
                gap: 15px;
                align-items: end;
            }
            .demo-buttons {
                display: flex;
                gap: 10px;
                margin: 15px 0;
                flex-wrap: wrap;
            }
            .demo-btn {
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.3s ease;
                width: auto;
            }
            .demo-btn:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-1px);
            }
            @media (max-width: 768px) {
                .flex-row {
                    grid-template-columns: 1fr;
                    gap: 10px;
                }
                .demo-buttons {
                    justify-content: center;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Emoji Recommender</h1>
            <p class="subtitle">Advanced AI models for intelligent emoji prediction</p>
            
            <div class="demo-buttons">
                <button class="demo-btn" onclick="setDemoText('Just received the best news ever!')">😊 Good News</button>
                <button class="demo-btn" onclick="setDemoText('I am so tired of this weather')">😩 Tired</button>
                <button class="demo-btn" onclick="setDemoText('My team absolutely crushed it today!')">🚀 Success</button>
                <button class="demo-btn" onclick="setDemoText('This traffic is killing me')">😤 Frustrated</button>
                <button class="demo-btn" onclick="setDemoText('Beautiful sunset today')">🌅 Beautiful</button>
            </div>
            
            <div class="input-group">
                <label for="text">Enter your text:</label>
                <textarea id="text" placeholder="Type something and get AI-powered emoji recommendations... (Ctrl+Enter to predict)"></textarea>
            </div>
            
            <div class="flex-row">
                <div class="input-group">
                    <label for="model">AI Model:</label>
                    <select id="model">
                        <option value="deepmoji">DeepMoji</option>
                        <option value="distilbert">DistilBERT</option>
                        <option value="mobilebert">MobileBERT</option>
                    </select>
                </div>
                
                <div class="input-group">
                    <label for="topk">Top K:</label>
                    <input type="number" id="topk" value="5" min="1" max="10">
                </div>
                
                <div class="input-group">
                    <button onclick="getRecommendations()" id="predictBtn">
                        🔮 Predict Emojis
                    </button>
                </div>
            </div>
            
            <div id="results" class="results">
                <h3>📊 AI Predictions:</h3>
                <div id="predictions"></div>
                <div id="modelInfo" class="model-info"></div>
            </div>
        </div>

        <script>
            function setDemoText(text) {
                document.getElementById('text').value = text;
            }
            
            async function getRecommendations() {
                const text = document.getElementById('text').value.trim();
                const model = document.getElementById('model').value;
                const topk = parseInt(document.getElementById('topk').value);
                const resultsDiv = document.getElementById('results');
                const predictionsDiv = document.getElementById('predictions');
                const modelInfoDiv = document.getElementById('modelInfo');
                const btn = document.getElementById('predictBtn');
                
                if (!text) {
                    alert('Please enter some text!');
                    return;
                }
                
                btn.disabled = true;
                btn.textContent = '🔄 AI is thinking...';
                resultsDiv.style.display = 'block';
                predictionsDiv.innerHTML = '<div class="loading">🧠 AI is analyzing your text...</div>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            top_k: topk,
                            model_type: model
                        })
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    predictionsDiv.innerHTML = '';
                    data.predictions.forEach((pred, index) => {
                        const predDiv = document.createElement('div');
                        predDiv.className = 'prediction';
                        predDiv.innerHTML = `
                            <span class="emoji">${pred.emoji}</span>
                            <div class="prediction-info">
                                <div class="rank">Rank ${pred.rank}</div>
                            </div>
                            <span class="confidence">${pred.confidence.toFixed(1)}%</span>
                        `;
                        predictionsDiv.appendChild(predDiv);
                    });
                    
                    modelInfoDiv.innerHTML = `
                        <strong>Model:</strong> ${data.model_used.toUpperCase()} | 
                        <strong>F1 Score:</strong> <span class="f1-score">${data.f1_score.toFixed(4)}</span> | 
                        <strong>Processing:</strong> ${(data.processing_time * 1000).toFixed(1)}ms
                    `;
                    
                } catch (error) {
                    predictionsDiv.innerHTML = `<div style="color: #ff6b6b; text-align: center; padding: 20px;">
                        <strong>❌ Error:</strong> ${error.message}
                    </div>`;
                } finally {
                    btn.disabled = false;
                    btn.textContent = '🔮 Predict Emojis';
                }
            }
            
            // Allow Enter key to trigger prediction
            document.getElementById('text').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    getRecommendations();
                }
            });
            
            // Load available models on page load
            async function loadModels() {
                try {
                    const response = await fetch('/models');
                    const models = await response.json();
                    const select = document.getElementById('model');
                    select.innerHTML = '';
                    
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.name;
                        option.textContent = `${model.display_name} (F1: ${model.f1_score.toFixed(4)})`;
                        select.appendChild(option);
                    });
                } catch (error) {
                    console.error('Failed to load models:', error);
                }
            }
            
            // Load models when page loads
            loadModels();
            
            // Set focus on text area
            document.getElementById('text').focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/models")
async def get_available_models():
    """Get list of available models with F1 scores"""
    if recommendation_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    return recommendation_system.get_available_models()

@app.post("/predict")
async def predict_emojis(input_data: TextInput):
    """Predict emojis for given text"""
    if recommendation_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    try:
        result = recommendation_system.recommend_emojis(
            input_data.text, 
            input_data.top_k, 
            input_data.model_type
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch_emojis(input_data: BatchTextInput):
    """Predict emojis for multiple texts"""
    if recommendation_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    try:
        results = recommendation_system.batch_recommend(
            input_data.texts, 
            input_data.top_k, 
            input_data.model_type
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(recommendation_system.models) if recommendation_system else 0,
        "available_models": [m['name'] for m in recommendation_system.get_available_models()] if recommendation_system else [],
        "device": str(recommendation_system.config.DEVICE) if recommendation_system else "unknown"
    }

@app.get("/compare/{text}")
async def compare_models(text: str, top_k: int = 3):
    """Compare predictions across all models for a given text"""
    if recommendation_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    results = {}
    for model_info in recommendation_system.get_available_models():
        try:
            result = recommendation_system.recommend_emojis(text, top_k, model_info['name'])
            results[model_info['name']] = result
        except Exception as e:
            results[model_info['name']] = {"error": str(e)}
    
    return {
        "text": text,
        "comparisons": results
    }

def test_system():
    """Test the recommendation system with real-time sentences"""
    if recommendation_system is None:
        print("❌ Recommendation system not initialized")
        return
    
    print("🤖 Emoji Recommendation System - Testing")
    print("=" * 60)
    
    # Get available models
    models_info = recommendation_system.get_available_models()
    print("📊 Available Models:")
    for i, model_info in enumerate(models_info, 1):
        print(f"  {i}. {model_info['display_name']}: F1 = {model_info['f1_score']:.4f}")
    
    if not models_info:
        print("❌ No models available for testing")
        return
    
    # Test sentences
    test_sentences = [
        "Just got promoted at work! So excited!",
        "I'm really tired after this long day",
        "The weather is absolutely beautiful today",
        "My team won the championship!",
        "This traffic is so frustrating"
    ]
    
    print("\n🧪 Testing with sample sentences:\n")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"🔍 Test {i}: '{sentence}'")
        print("-" * 40)
        
        for model_info in models_info:
            model_name = model_info['name']
            try:
                result = recommendation_system.recommend_emojis(
                    sentence, top_k=3, model_type=model_name
                )
                
                emojis = [p['emoji'] for p in result['predictions']]
                confidences = [f"{p['confidence']:.1f}%" for p in result['predictions']]
                
                print(f"  {model_name.upper()}: {' '.join(emojis)} | {' '.join(confidences)} | {result['processing_time']*1000:.1f}ms")
                
            except Exception as e:
                print(f"  {model_name.upper()}: ❌ Error - {e}")
        
        print()
    
    print("✅ Testing completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Emoji Recommendation System")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    
    args = parser.parse_args()
    
    if args.test:
        test_system()
    else:
        print("🚀 Starting Emoji Recommendation API Server...")
        print(f"🌐 UI available at: http://{args.host}:{args.port}")
        print(f"📖 API docs at: http://{args.host}:{args.port}/docs")
        
        if recommendation_system:
            models = recommendation_system.get_available_models()
            print(f"🤖 Loaded {len(models)} models:")
            for model in models:
                print(f"   • {model['display_name']}: F1 = {model['f1_score']:.4f}")
        
        uvicorn.run(
            "recommender:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level="info"
        )