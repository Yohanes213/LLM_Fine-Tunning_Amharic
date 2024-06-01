from preprocess import normalize_char_level_missmatch, clean_document

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

  label_to_category = { 0: 'Others',
                        1: 'Politics',
                        2: 'International News',
                        3: 'Local News',
                        4: 'Entertainment',
                        5: 'Sports',
                        6: 'Business'}

  return f"This news belongs to {label_to_category[predicted_label]}"

