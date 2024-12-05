from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re  

app = Flask(__name__)
CORS(app)

# Load the dataset
dataset = pd.read_csv('C:/Users/rahma/Documents/S2/ITFFC_M/model/text_data.csv')
print(f"Dataset loaded with {len(dataset)} entries")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./model")

def is_in_dataset(text):
    """Check if the input text matches any entry in the dataset."""
    row = dataset[dataset['text'].str.strip() == text.strip()]
    if not row.empty:
        label = int(row['label'].values[0])
        return "Human-Written" if label == 0 else "AI-Generated"
    return None  # If text isn't found in the dataset

def is_valid_text(text):
    """Basic validation to check if the input text is meaningful."""
    if len(text.strip()) == 0 or not re.search(r'\w+', text):
        return False
    return True

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the text classification API!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({"error": "No content provided"}), 400

        text = data['content']

        
        if not is_valid_text(text):
            return jsonify({"error": "Invalid or nonsensical text provided"}), 400

        
        dataset_result = is_in_dataset(text)
        if dataset_result:
            return jsonify({"result": dataset_result})

        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.softmax(outputs.logits, dim=1)

        
        ai_generated_prob = prediction[0][1].item()
        result = "AI-Generated" if ai_generated_prob > 0.5 else "Human-Written"
        
        return jsonify({"result": result})

    except Exception as e:
        print("An error occurred:", e)
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
