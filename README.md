📰 News Article Summarizer, Sentiment & Bias Detector

A Streamlit app that summarizes news articles, analyzes sentiment using RoBERTa, and detects political bias with zero-shot classification. Supports text and image input (OCR). Built with transformers, torch, and HuggingFace models for real-time NLP insights.

---

🚀 Features

- 🔤 **Text & Image Input**: Paste text or upload a news article image (`.png`, `.jpg`, `.jpeg`).
- 📄 **OCR Extraction**: Extracts readable text from images using Tesseract OCR.
- 📝 **Summarization**: Uses Azure-hosted GPT-4.1 (via `azure.ai.inference`) to generate short, meaningful summaries.
- ❤️ **Sentiment Analysis**: Identifies sentiment as *Positive*, *Negative*, or *Neutral* using a custom-trained model.
- ⚖️ **Bias Detection**: Performs zero-shot classification to detect political bias (*Left*, *Right*, *Neutral*) and visualize confidence scores with interactive charts.
- 📊 **Altair Visualizations**: Clean bar charts to show model confidence for bias detection.
---
🧠 Tech Stack

| Layer       | Technology              |
|-------------|--------------------------|
| UI          | Streamlit                |
| Image OCR   | Tesseract (`pytesseract`)|
| LLM         | Azure OpenAI GPT-4.1     |
| Visualization | Altair + Pandas       |
| Model Access | azure.ai.inference SDK |
| Text Processing | Custom Python models in `biased_modell.py` |
---
📁 Folder Structure
.
├── biased\_modell.py          # Custom logic for sentiment & bias detection
├── streamlit\_app.py          # Streamlit app entry point
├── requirements.txt          # All project dependencies
├── .gitignore
├── .env                      
└── README.md                 # Project documentation
---
⚙️ Setup Instructions

🔧 Prerequisites

- Python 3.8+
- Tesseract installed and added to your system PATH ([Install Guide](https://github.com/tesseract-ocr/tesseract))

📦 Installation

```bash
# Clone the repo
git clone https://github.com/gayathripavushetty18/Bias-and-Sentimental-analyzer.git
cd Bias-and-Sentimental-analyzer

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
````

🔐 Set Up Your Environment

Create a `.env` file and add your GitHub token:

```
GITHUB_TOKEN=your_github_token_here
```
## ▶️ Run the App

```bash
streamlit run streamlit_app.py
```

Then open the provided local URL in your browser.

---

## 📌 To-Do (Future Work)

* Multilingual article support (summarization + bias)
* Model fine-tuning options via UI
* Support uploading PDFs or entire URLs

---

🙌 Contributing
Pull requests and issues are welcome! If you want to improve the analysis models or UI, feel free to fork and contribute.
📜 License
This project is licensed under the [MIT License](LICENSE).
---
✨ Acknowledgments
* [Streamlit](https://streamlit.io/)
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
* [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai)
* HuggingFace Transformers (used inside `biased_modell.py`)
---
👩‍💻 Author
**Gayathri Pavushetty**
Feel free to connect on [GitHub](https://github.com/gayathripavushetty18)
---
