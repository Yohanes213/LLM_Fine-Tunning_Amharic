from preprocess import normalize_char_level_missmatch, clean_document

def preprocess_article(row):
    article = row['article']
    article = normalize_char_level_missmatch(article)
    article = clean_document(article)
    return {'article': article}

def calculate_length(row):
  article = row['article']
  word_count = len(article)
  return {'word_count': word_count}