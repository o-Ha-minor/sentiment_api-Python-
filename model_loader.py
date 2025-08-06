# model_loader.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
