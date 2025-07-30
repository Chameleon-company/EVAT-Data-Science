from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("saved_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")
model.eval()

# Create Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    labels = ["negative", "neutral", "positive"]

    return jsonify({
        "text": text,
        "prediction": labels[predicted_class],
        "confidence": round(confidence, 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
