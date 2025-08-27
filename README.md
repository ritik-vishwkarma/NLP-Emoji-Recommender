# 🎯 NLP Emoji Recommender

AI-powered emoji prediction system for text!  
Built with FastAPI, PyTorch, and Transformers.  
Ensemble of finetuned models for accurate emoji recommendations.

---

![Demo Screenshot](images/image.png)

---

## 🚀 Features

- **Ensemble Prediction:** Combines multiple finetuned models for best results.
- **FastAPI Web App:** Modern, responsive UI for interactive emoji recommendations.
- **REST API:** `/predict` endpoint for programmatic access.
- **Configurable:** Choose number of emojis and confidence threshold.
- **Health & Model Info Endpoints:** For monitoring and debugging.

---

## 🖥️ Demo

![Demo Screenshot](images/image.png)

---

## 📦 Installation

```bash
git clone https://github.com/ritik-vishwkarma/NLP-Emoji-Recommender.git
cd NLP-Emoji-Recommender
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚡ Usage

1. **Download or place your finetuned model files** in `models_advanced/` directory.
2. **Run the recommender:**
   ```bash
   python recommender.py
   ```
3. **Open your browser:**  
   [http://localhost:8000](http://localhost:8000)
4. **API Docs:**  
   [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📝 Requirements

```
fastapi
uvicorn
torch>=2.0
transformers
numpy
pydantic
```

---

## 📚 API Endpoints

- `/` : Web UI
- `/predict` : POST, returns emoji recommendations for input text
- `/health` : Health check
- `/models` : Model info

---

## 🛠️ Model Files

Place your finetuned model files in `models_advanced/`:
- `best_distilbert_f1_0.4205.pth`
- `best_multihead_f1_0.4010.pth`
- (Optional) `best_deepmoji_f1_0.4828.pth`

---

## 🤝 Contributing

Pull requests and issues welcome!

---

## 📄 License

MIT

---

**Repo:** [https://github.com/ritik-vishwkarma/NLP-Emoji-Recommender](https://github.com/ritik-vishwkarma/NLP-Emoji-Recommender)