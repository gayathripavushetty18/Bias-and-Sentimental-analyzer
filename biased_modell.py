import re
import torch
import pandas as pd
from transformers import pipeline

# -------------------- Device Configuration --------------------
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU
print("Device set to use", "GPU" if device == 0 else "CPU")

# -------------------- Model Names --------------------
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"

# -------------------- Text Cleaning --------------------
def clean_text(text):
    """Clean and normalize text by removing extra whitespace and line breaks."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text.strip()

# -------------------- Text Chunking --------------------
def split_text(text, max_length=512):
    """Split long text into chunks for model processing."""
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_length:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -------------------- Pipelines with Device --------------------
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME, device=device)
zero_shot_pipeline = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL_NAME, device=device)

# -------------------- Sentiment Analysis --------------------
LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

def get_final_sentiment(text, sentiment_pipe=sentiment_pipeline):
    cleaned_text = clean_text(text)
    chunks = split_text(cleaned_text)
    sentiment_scores = []

    for chunk in chunks:
        try:
            result = sentiment_pipe(chunk)
            raw_label = result[0]['label']
            readable_label = LABEL_MAP.get(raw_label, raw_label)
            sentiment_scores.append(readable_label)
        except Exception as e:
            print("Sentiment analysis failed on chunk:", e)

    if sentiment_scores:
        final_sentiment = max(set(sentiment_scores), key=sentiment_scores.count)
        return final_sentiment
    return "Unknown"

# -------------------- Zero-Shot Bias Detection --------------------
def zero_shot_bias(text):
    cleaned_text = clean_text(text)
    candidate_labels = ["left-bias", "right-bias", "neutral"]

    try:
        result = zero_shot_pipeline(cleaned_text, candidate_labels)
        return {
            "Bias_label": result['labels'][0],
            "Bias_scores": dict(zip(result['labels'], result['scores']))
        }
    except Exception as e:
        print("Bias detection failed:", e)
        return {
            "Bias_label": "Unknown",
            "Bias_scores": {}
        }

# -------------------- Bias Model Accuracy Calculation --------------------
def calculate_bias_model_accuracy(test_csv_path):
    df = pd.read_csv(test_csv_path)
    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        prediction = zero_shot_bias(row['article'])
        predicted_label = prediction['Bias_label']

        # Derive true label from highest score column
        score_map = {
            "left-bias": row['left_bias_score'],
            "right-bias": row['right_bias_score'],
            "neutral": row['neutral_bias_score']
        }
        true_label = max(score_map, key=score_map.get)

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Bias Model Accuracy on test set: {accuracy * 100:.2f}%")
    return accuracy

# -------------------- Main --------------------
if __name__ == "__main__":
    test_csv = "E:/Summer Internship/bias_dataset_final.csv"

    calculate_bias_model_accuracy(test_csv)

    calculate_bias_model_accuracy(test_csv)