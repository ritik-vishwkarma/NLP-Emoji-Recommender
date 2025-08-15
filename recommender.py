import torch
from transformers import AutoTokenizer
import json
from models import MobileBERTEmojiClassifier, DeepMojiEmojiClassifier, EmojiPredictor
from config import Config

class EmojiRecommendationSystem:
    def __init__(self, model_type='mobilebert', model_path=None):
        self.config = Config()
        self.model_type = model_type.lower()
        
        # Load emoji mappings
        with open('models/emoji_mappings.json', 'r') as f:
            mappings = json.load(f)
            self.emoji_to_id = mappings['emoji_to_id']
            self.id_to_emoji = {int(k): v for k, v in mappings['id_to_emoji'].items()}
        
        # Initialize model and tokenizer
        if self.model_type == 'mobilebert':
            self.model = MobileBERTEmojiClassifier(len(self.emoji_to_id))
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MOBILEBERT_MODEL_NAME)
            model_path = model_path or 'models/mobilebert_final.pth'
        else:  # deepmoji
            self.model = DeepMojiEmojiClassifier(len(self.emoji_to_id))
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.DEEPMOJI_MODEL_NAME)
            model_path = model_path or 'models/deepmoji_final.pth'
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create predictor
        self.predictor = EmojiPredictor(
            self.model, self.tokenizer, self.emoji_to_id, self.id_to_emoji, self.config.DEVICE
        )
    
    def recommend_emojis(self, text, top_k=5):
        """Recommend emojis for given text"""
        return self.predictor.predict(text, top_k)
    
    def batch_recommend(self, texts, top_k=5):
        """Recommend emojis for multiple texts"""
        return self.predictor.predict_batch(texts, top_k)

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = EmojiRecommendationSystem('mobilebert')
    
    # Test examples
    test_texts = [
        "Just received the best news ever!",
        "I'm so tired of this heatwave.",
        "My team absolutely crushed it today.",
        "The new update is buggy.",
        "That concert was fire!",
        "I'm literally crying."
    ]
    
    print("🤖 Emoji Recommendation System Demo")
    print("=" * 50)
    
    for text in test_texts:
        recommendations = system.recommend_emojis(text, top_k=3)
        emojis = [r['emoji'] for r in recommendations]
        confidences = [f"{r['confidence']:.1f}%" for r in recommendations]
        
        print(f"Text: '{text}'")
        print(f"Emojis: {' '.join(emojis)}")
        print(f"Confidence: {' '.join(confidences)}")
        print()