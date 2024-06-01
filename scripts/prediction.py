from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from preprocess import normalize_char_level_missmatch, clean_document
from preprocess import normalize_char_level_missmatch, clean_document


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    result = prediction(text)
    return jsonify({'prediction': result})

def prediction(text):

  tokenizer = AutoTokenizer.from_pretrained('Yohanes213/Amharic_news_classification')
  model = AutoModelForSequenceClassification.from_pretrained('Yohanes213/Amharic_news_classification')


  text = normalize_char_level_missmatch(text)
  text = clean_document(text)

  inputs= tokenizer(text, return_tensors="pt")

  outputs = model(**inputs)

  probabilities = torch.softmax(outputs.logits, dim=1)

  # Get the predicted label (index with highest probability)
  predicted_label = torch.argmax(probabilities, dim=1).item()

  label_to_category = { 0: 'ሌሎች',
                        1: 'ፖለቲካ',
                        2: 'አለማቀፍ',
                        3: 'ሃገር ዉስጥ',
                        4: 'መዝናኛ',
                        5: 'ስፖርት',
                        6: 'ቢዝነስ'}

  return f"ይህ የ{label_to_category[predicted_label] } ዜና ነዉ::"
if __name__ == '__main__':
   app.run(host='localhost', port=5001)



