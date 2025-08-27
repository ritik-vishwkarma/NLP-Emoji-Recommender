from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer
import json
import os
from typing import List, Dict, Optional
import uvicorn

# Import your model classes
from src.models import DistilBERTEmojiClassifier, DeepMojiEmojiClassifier, MultiHeadEmojiClassifier

app = FastAPI(title="Emoji Recommender API", description="AI-powered emoji prediction system")

class PredictionRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5
    confidence_threshold: Optional[float] = 0.3

class EmojiRecommender:
    def __init__(self):
        # Simple config for recommender - no need for training config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 128
        self.models = {}
        self.tokenizers = {}
        self.emoji_mapping = None
        self.id_to_emoji = None
        
        # Model configurations
        self.model_configs = {
            'distilbert': {
                'class': DistilBERTEmojiClassifier,
                'model_path': 'distilbert-base-uncased',
                'file_pattern': 'best_distilbert_f1_0.4205.pth'
            },
            'deepmoji': {
                'class': DeepMojiEmojiClassifier,
                'model_path': 'cardiffnlp/twitter-roberta-base-emoji',
                'file_pattern': 'best_deepmoji_f1_0.4828.pth'
            },
            'multihead': {
                'class': MultiHeadEmojiClassifier,
                'model_path': 'distilbert-base-uncased',
                'file_pattern': 'best_multihead_f1_0.4010.pth'
            }
        }
        
        print("🚀 Initializing Emoji Recommender...")
        self.load_emoji_mapping()
        self.load_models()
    
    def load_emoji_mapping(self):
        """Load emoji mapping from a JSON file or create default mapping"""
        mapping_file = 'emoji_mapping.json'
        
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    self.emoji_mapping = json.load(f)
                self.id_to_emoji = {v: k for k, v in self.emoji_mapping.items()}
                print(f"✅ Loaded emoji mapping with {len(self.emoji_mapping)} emojis")
                return
            except Exception as e:
                print(f"⚠️  Error loading emoji mapping: {e}")
        
        # Create default mapping for common emojis
        default_emojis = [
            "😀", "😃", "😄", "😁", "😆", "😅", "🤣", "😂", "🙂", "🙃",
            "😉", "😊", "😇", "🥰", "😍", "🤩", "😘", "😗", "😚", "😙",
            "😋", "😛", "😜", "🤪", "😝", "🤑", "🤗", "🤭", "🤫", "🤔",
            "😐", "😑", "😶", "🙄", "😏", "😣", "😥", "😮", "🤐", "😯",
            "😪", "😫", "😴", "😌", "😒", "😓", "😔", "😕", "😲", "☹️", 
            "🙁", "😖", "😞", "😟", "😤", "😢", "😭", "😦", "😧", "😨", 
            "😩", "🤯", "😬", "😰", "😱", "🥵", "🥶", "😳", "😵", "😡", 
            "😠", "🤬", "😷", "🤒", "🤕", "🤢", "🤮", "🤧", "🤠", "🥳", 
            "😎", "🤓", "🧐", "🥺", "🥱", "😈", "👿", "👹", "👺", "💀",
            "☠️", "👻", "👽", "🤖", "💩", "😺", "😸", "😹", "😻", "😼",
            "😽", "🙀", "😿", "😾", "🐶", "🐱", "🐭", "🐹", "🐰", "🦊"
        ]
        
        self.emoji_mapping = {emoji: i for i, emoji in enumerate(default_emojis)}
        self.id_to_emoji = {i: emoji for emoji, i in self.emoji_mapping.items()}
        
        # Save the mapping
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.emoji_mapping, f, ensure_ascii=False, indent=2)
            print(f"✅ Created default emoji mapping with {len(self.emoji_mapping)} emojis")
        except Exception as e:
            print(f"⚠️  Could not save emoji mapping: {e}")

    def load_models(self):
        """Load the three best trained models"""
        model_dir = 'models_advanced'
        
        if not os.path.exists(model_dir):
            print(f"❌ Model directory '{model_dir}' not found!")
            print("🔍 Available directories:", [d for d in os.listdir('.') if os.path.isdir(d)])
            raise RuntimeError(f"Model directory '{model_dir}' not found!")
        
        print(f"📁 Looking for models in: {model_dir}")
        available_files = os.listdir(model_dir)
        print(f"📄 Available files: {available_files}")
        
        for model_name, config in self.model_configs.items():
            try:
                # Find model file
                model_files = [f for f in available_files if config['file_pattern'] in f]
                if not model_files:
                    print(f"⚠️  Model file matching '{config['file_pattern']}' not found, skipping {model_name}...")
                    continue
                
                model_file = model_files[0]
                model_path = os.path.join(model_dir, model_file)
                print(f"🔄 Loading {model_name} from {model_file}...")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                self.tokenizers[model_name] = tokenizer
                
                # Load saved weights to determine number of classes
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Determine number of classes from the loaded weights
                num_classes = None
                classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
                
                if classifier_keys:
                    # For DeepMoji and other models, find the final classification layer
                    if model_name == 'deepmoji':
                        # Look for the final layer in the classifier
                        final_layer_keys = [k for k in classifier_keys if k.endswith('.weight') and len(state_dict[k].shape) == 2]
                        if final_layer_keys:
                            # Sort to get the last layer (highest number)
                            final_key = sorted(final_layer_keys, key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 0)[-1]
                            num_classes = state_dict[final_key].shape[0]
                    else:
                        # For other models, use the first classifier weight found
                        for key in classifier_keys:
                            if len(state_dict[key].shape) == 2:
                                num_classes = state_dict[key].shape[0]
                                break
                
                if num_classes is None:
                    print(f"⚠️  Could not determine number of classes for {model_name}, using default 100")
                    num_classes = 100
                
                print(f"🔍 Detected {num_classes} classes for {model_name}")
                
                # Initialize model with correct number of classes
                if model_name == 'deepmoji':
                    model = config['class'](num_classes=num_classes, model_name=config['model_path'])
                else:
                    model = config['class'](num_classes=num_classes, model_name=config['model_path'])
                
                # Load the state dict with error handling
                try:
                    model.load_state_dict(state_dict, strict=True)
                except RuntimeError as e:
                    print(f"⚠️  Strict loading failed for {model_name}, trying non-strict loading...")
                    model.load_state_dict(state_dict, strict=False)
                    
                model.to(self.device)
                model.eval()
                
                self.models[model_name] = model
                print(f"✅ Loaded {model_name} model with {num_classes} classes")
                
            except Exception as e:
                print(f"❌ Error loading {model_name}: {str(e)}")
                print(f"🔍 Skipping {model_name} and continuing...")
                continue
        
        if not self.models:
            raise RuntimeError("❌ No models could be loaded! Check model files and paths.")
        
        print(f"🎯 Successfully loaded {len(self.models)} models: {list(self.models.keys())}")

    
    def preprocess_text(self, text: str, tokenizer):
        """Preprocess text for model input"""
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict_single_model(self, text: str, model_name: str):
        """Get predictions from a single model"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Preprocess
        inputs = self.preprocess_text(text, tokenizer)
        
        # Predict
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.sigmoid(logits)
        
        return probabilities.cpu().numpy()[0]
    
    def predict_ensemble(self, text: str, top_k: int = 5, confidence_threshold: float = 0.3):
        """Get ensemble predictions from all models"""
        if not text.strip():
            return []
        
        all_predictions = []
        # Model weights based on their F1 scores
        model_weights = {
            'deepmoji': 0.4828,    # Best performing
            'distilbert': 0.4205,  # Second best
            'multihead': 0.4010    # Third best
        }
        
        # Normalize weights
        total_weight = sum(model_weights.get(name, 0) for name in self.models.keys())
        if total_weight > 0:
            model_weights = {name: weight/total_weight for name, weight in model_weights.items()}
        else:
            # Equal weights if no predefined weights
            model_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        
        # Get predictions from each model
        for model_name in self.models.keys():
            try:
                pred = self.predict_single_model(text, model_name)
                if pred is not None:
                    weight = model_weights.get(model_name, 1.0 / len(self.models))
                    all_predictions.append(pred * weight)
                    print(f"🔮 {model_name}: weight={weight:.3f}, max_prob={pred.max():.3f}")
            except Exception as e:
                print(f"❌ Error predicting with {model_name}: {e}")
                continue
        
        if not all_predictions:
            return []
        
        # Ensemble average
        ensemble_pred = np.sum(all_predictions, axis=0)
        
        # Get top predictions
        top_indices = np.argsort(ensemble_pred)[::-1]
        
        results = []
        for idx in top_indices[:top_k * 3]:  # Get more candidates
            confidence = float(ensemble_pred[idx])
            if confidence >= confidence_threshold and len(results) < top_k:
                emoji = self.id_to_emoji.get(idx, f"emoji_{idx}")
                results.append({
                    'emoji': emoji,
                    'confidence': confidence,
                    'emoji_id': int(idx)
                })
        
        print(f"🎯 Ensemble prediction: {len(results)} emojis above threshold {confidence_threshold}")
        return results

# Initialize recommender
print("🔥 Starting Emoji Recommender initialization...")
recommender = EmojiRecommender()

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Simple web interface for emoji prediction"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎯 Emoji Recommender</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { text-align: center; margin-bottom: 30px; }
            textarea { 
                width: 100%; 
                height: 100px; 
                padding: 15px; 
                border: none; 
                border-radius: 10px; 
                font-size: 16px;
                resize: vertical;
                box-sizing: border-box;
            }
            button { 
                background: #ff6b6b; 
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 10px; 
                cursor: pointer; 
                font-size: 16px;
                margin-top: 15px;
                width: 100%;
            }
            button:hover { background: #ff5252; }
            .results { 
                margin-top: 20px; 
                padding: 20px; 
                background: rgba(255, 255, 255, 0.1); 
                border-radius: 10px; 
                min-height: 100px;
            }
            .emoji-result { 
                display: inline-block; 
                margin: 10px; 
                padding: 15px; 
                background: rgba(255, 255, 255, 0.2); 
                border-radius: 10px; 
                text-align: center;
                min-width: 80px;
            }
            .emoji { font-size: 32px; margin-bottom: 5px; }
            .confidence { font-size: 12px; opacity: 0.8; }
            .loading { text-align: center; opacity: 0.7; }
            select { 
                width: 100%; 
                padding: 10px; 
                border: none; 
                border-radius: 5px; 
                font-size: 14px;
                box-sizing: border-box;
            }
            label { 
                display: block; 
                margin-bottom: 5px; 
                font-size: 14px; 
                opacity: 0.9; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 AI Emoji Recommender</h1>
            <p style="text-align: center; opacity: 0.9;">Enter your text and get AI-powered emoji recommendations!</p>
            
            <textarea id="textInput" placeholder="Type your message here... (e.g., 'I'm so happy today!' or 'This is amazing!')"></textarea>
            
            <div style="display: flex; gap: 15px; margin-top: 15px;">
                <div style="flex: 1;">
                    <label>Number of Emojis:</label>
                    <select id="topK">
                        <option value="3">3</option>
                        <option value="5" selected>5</option>
                        <option value="8">8</option>
                        <option value="10">10</option>
                    </select>
                </div>
                <div style="flex: 1;">
                    <label>Confidence Threshold:</label>
                    <select id="confidence">
                        <option value="0.1">0.1 (More emojis)</option>
                        <option value="0.2">0.2</option>
                        <option value="0.3" selected>0.3 (Balanced)</option>
                        <option value="0.4">0.4</option>
                        <option value="0.5">0.5 (High confidence)</option>
                    </select>
                </div>
            </div>
            
            <button onclick="predictEmojis()">🔮 Predict Emojis</button>
            
            <div id="results" class="results">
                <div class="loading">Enter text above and click "Predict Emojis" to see recommendations! 🚀</div>
            </div>
        </div>

        <script>
            async function predictEmojis() {
                const text = document.getElementById('textInput').value;
                const topK = document.getElementById('topK').value;
                const confidence = document.getElementById('confidence').value;
                const resultsDiv = document.getElementById('results');
                
                if (!text.trim()) {
                    resultsDiv.innerHTML = '<div class="loading">Please enter some text first! 📝</div>';
                    return;
                }
                
                resultsDiv.innerHTML = '<div class="loading">🤖 AI is analyzing your text... Please wait!</div>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            top_k: parseInt(topK),
                            confidence_threshold: parseFloat(confidence)
                        }),
                    });
                    
                    const data = await response.json();
                    
                    if (data.predictions && data.predictions.length > 0) {
                        let html = '<h3>🎯 Recommended Emojis:</h3>';
                        data.predictions.forEach((pred, index) => {
                            html += `
                                <div class="emoji-result">
                                    <div class="emoji">${pred.emoji}</div>
                                    <div class="confidence">${(pred.confidence * 100).toFixed(1)}%</div>
                                </div>
                            `;
                        });
                        html += `<p style="margin-top: 15px; opacity: 0.7; font-size: 12px;">
                                Processed by ${data.model_count} AI models</p>`;
                        resultsDiv.innerHTML = html;
                    } else {
                        resultsDiv.innerHTML = '<div class="loading">😅 No confident emoji predictions found. Try different text or lower the confidence threshold!</div>';
                    }
                } catch (error) {
                    resultsDiv.innerHTML = '<div class="loading">❌ Error occurred. Please try again!</div>';
                    console.error('Error:', error);
                }
            }
            
            // Allow Enter key to trigger prediction
            document.getElementById('textInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    predictEmojis();
                }
            });
            
            // Add some example texts
            function setExample(text) {
                document.getElementById('textInput').value = text;
            }
            
            // Add example buttons after page loads
            window.onload = function() {
                const container = document.querySelector('.container');
                const exampleDiv = document.createElement('div');
                exampleDiv.innerHTML = `
                    <h4 style="margin-top: 20px; opacity: 0.8;">Try these examples:</h4>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px;">
                        <button onclick="setExample('I love this so much!')" style="flex: none; width: auto; padding: 8px 15px; font-size: 12px;">😍 Love</button>
                        <button onclick="setExample('This is hilarious!')" style="flex: none; width: auto; padding: 8px 15px; font-size: 12px;">😂 Funny</button>
                        <button onclick="setExample('I am so tired today')" style="flex: none; width: auto; padding: 8px 15px; font-size: 12px;">😴 Tired</button>
                        <button onclick="setExample('What an amazing day!')" style="flex: none; width: auto; padding: 8px 15px; font-size: 12px;">🌟 Amazing</button>
                    </div>
                `;
                container.insertBefore(exampleDiv, document.getElementById('results'));
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_emojis(request: PredictionRequest):
    """API endpoint for emoji prediction"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        predictions = recommender.predict_ensemble(
            text=request.text,
            top_k=request.top_k,
            confidence_threshold=request.confidence_threshold
        )
        
        return {
            "text": request.text,
            "predictions": predictions,
            "model_count": len(recommender.models),
            "available_models": list(recommender.models.keys()),
            "total_emojis": len(recommender.emoji_mapping) if recommender.emoji_mapping else 0
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(recommender.models),
        "available_models": list(recommender.models.keys()),
        "total_emojis": len(recommender.emoji_mapping) if recommender.emoji_mapping else 0,
        "device": str(recommender.device)
    }

@app.get("/models")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "loaded_models": list(recommender.models.keys()),
        "model_configs": recommender.model_configs,
        "device": str(recommender.device),
        "emoji_count": len(recommender.emoji_mapping) if recommender.emoji_mapping else 0
    }

if __name__ == "__main__":
    print("🚀 Starting Emoji Recommender API...")
    print("🌐 Open http://localhost:8000 in your browser")
    print("📚 API docs available at http://localhost:8000/docs")
    try:
        uvicorn.run("recommender:app", host="0.0.0.0", port=8000, reload=True)
    except:
        # Fallback without reload
        uvicorn.run(app, host="0.0.0.0", port=8000)