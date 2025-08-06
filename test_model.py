# test_sentiment.py
from model_loader import tokenizer, model
import torch

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment = 'positive' if probs[0][1] > probs[0][0] else 'negative'
    score = round(probs[0][1].item(), 4)
    return sentiment, score

# テスト文
test_sentences = [
    "心の雨が降り止まない", 
    "ああ、どうか安らかに眠れ",
    "もう何も悩みはなく、ただ眠りたい",
    "もう疲れた"
]

for text in test_sentences:
    sentiment, score = analyze_sentiment(text)
    print(f"文章: {text} → 感情: {sentiment}, スコア: {score}")
