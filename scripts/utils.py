from scripts.preprocess import normalize_char_level_missmatch, clean_document
import evaluate
import numpy as np
from torch.utils.data import DataLoader
import torch
import pandas as pd

def preprocess_article(row):
    article = row['article']
    article = normalize_char_level_missmatch(article)
    article = clean_document(article)
    return {'article': article}

def calculate_length(row):
    article = row['article']
    word_count = len(article)
    return {'word_count': word_count}

def tokenize_function(example, tokenizer, category_to_id):
    inputs = tokenizer(example['article'], padding=True, truncation=True, max_length=512)
    inputs["labels"] = category_to_id[example["category"]]
    return inputs

def compute_metrics(eval_preds):
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("precision")
    metric3 = evaluate.load("recall")
    metric4 = evaluate.load("f1")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric2.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    recall = metric3.compute(predictions=predictions, references=labels, average='weighted')["recall"]
    f1 = metric4.compute(predictions=predictions, references=labels, average='weighted')["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_metrics_eval(y_pred, y_test):
  metric1 = evaluate.load("accuracy")
  metric2 = evaluate.load("precision")
  metric3 = evaluate.load("recall")
  metric4 = evaluate.load("f1")

  #logits, labels = y_preds
  #predictions = np.argmax(logits, axis=-1)

  accuracy = metric1.compute(predictions=y_pred, references=y_test)["accuracy"]
  precision = metric2.compute(predictions=y_pred, references=y_test, average='weighted')["precision"]
  recall = metric3.compute(predictions=y_pred, references=y_test, average='weighted')["recall"]
  f1 = metric4.compute(predictions=y_pred, references=y_test, average='weighted')["f1"]

  return {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1": f1
  }



def evaluate_model(model, tokenized_datasets, data_collator, device):
    eval_dataset = tokenized_datasets["test"].remove_columns(['article', 'category', 'word_count']).with_format("torch")
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)

    y_test, y_pred = [], []
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_pred.extend(predictions.cpu().numpy())
        y_test.extend(batch["labels"].cpu().numpy())

    return compute_metrics_eval(y_pred, y_test)


def generate_predictions(model, tokenized_datasets, device, id_to_category, num_samples=5):
    # Assuming you have already defined your DataLoader for test data
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=1)

    # Set the model to evaluation mode
    model.eval()

    # Define a list to store predictions
    predictions = []

    # Loop through the specified number of data points and make predictions
    for i, batch in enumerate(test_dataloader):
        if i >= num_samples:
            break

        # Move batch to appropriate device (CPU or GPU)
        batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}

        with torch.no_grad():
            # Forward pass
            outputs = model(**batch)

        # Get the predicted probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)

        # Get the predicted label (index with highest probability)
        predicted_label = torch.argmax(probabilities, dim=1).item()

        # Append the predicted label to the list
        predictions.append(predicted_label)

    # Prepare data for DataFrame
    data = {'article':[],
                  'predicted_labels':[],
                  'true_labels': [],
                  'true_category': [],
                  'predicted_category':[]}
    for i in range(len(predictions)):
        data['article'].append(tokenized_datasets['test'][i]['article'])
        data['predicted_labels'].append(predictions[i])
        data['true_labels'].append(tokenized_datasets['test'][i]['labels'])
        data['true_category'].append(tokenized_datasets['test'][i]['category'])
        data['predicted_category'].append(id_to_category[predictions[i]])

    # Return DataFrame with predictions
    return pd.DataFrame(data)