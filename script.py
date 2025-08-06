# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS 
from model_loader import tokenizer, model
import torch


# モデルの感情判定表
id2label = {
    0: "neutral",
    1: "negative",
    2: "positive"
}

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Flask API is running. Use /analyze endpoint."

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400

        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        prediction = probs.argmax(dim=1).item()
        label = id2label[prediction]
        score = round(probs[0][prediction].item(), 4)

        return jsonify({
            'label': label,
            'score': score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(debug=True)
    
