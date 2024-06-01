from scripts.preprocess import normalize_char_level_missmatch, clean_document
import evaluate
import numpy as np
from torch.utils.data import DataLoader
import torch
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('preprocessing.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def preprocess_article(row):
    """
    Preprocess an article by normalizing character level mismatches and cleaning the text.

    Args:
        row (dict): A dictionary containing the article text under the key 'article'.

    Returns:
        dict: A dictionary with the preprocessed article text.
    """
    try:
        article = row['article']
        article = normalize_char_level_missmatch(article)
        article = clean_document(article) 
        return {'article': article}
    except Exception as e:
        logger.error(f"Error in preprocess_article: {e}")
        raise

def calculate_length(row):
    """
    Calculate the length of the article in words.

    Args:
        row (dict): A dictionary containing the article text under the key 'article'.

    Returns:
        dict: A dictionary with the word count of the article.
    """
    try:
        article = row['article']
        word_count = len(article)
        return {'word_count': word_count}
    except Exception as e:
        logger.error(f"Error in calculate_length: {e}")
        raise

def tokenize_function(example, tokenizer, category_to_id):
    """
    Tokenize an article and convert its category to an ID.

    Args:
        example (dict): A dictionary containing the article text under the key 'article' and its category under the key 'category'.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing the article.
        category_to_id (dict): A dictionary mapping category names to category IDs(labels).

    Returns:
        dict: A dictionary with tokenized inputs and the corresponding category ID as labels.
    """
    try:
        inputs = tokenizer(example['article'], padding=True, truncation=True, max_length=512)
        inputs["labels"] = category_to_id[example["category"]]
        return inputs
    except Exception as e:
        logger.error(f"Error in tokenize_function: {e}")
        raise

def compute_metrics(eval_preds):
    """
    Compute evaluation metrics including accuracy, precision, recall, and F1 score.

    Args:
        eval_preds (tuple): A tuple containing the logits and the true labels.

    Returns:
        dict: A dictionary with the computed metrics.
    """
    try:
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

        logger.info(f"Computed metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    except Exception as e:
        logger.error(f"Error in compute_metrics: {e}")
        raise

def compute_metrics_eval(y_pred, y_test):
    """
    Compute evaluation metrics for predictions against true labels.

    Args:
        y_pred (list): A list of predicted labels.
        y_test (list): A list of true labels.

    Returns:
        dict: A dictionary with the computed metrics.
    """
    try:
        metric1 = evaluate.load("accuracy")
        metric2 = evaluate.load("precision")
        metric3 = evaluate.load("recall")
        metric4 = evaluate.load("f1")

        accuracy = metric1.compute(predictions=y_pred, references=y_test)["accuracy"]
        precision = metric2.compute(predictions=y_pred, references=y_test, average='weighted')["precision"]
        recall = metric3.compute(predictions=y_pred, references=y_test, average='weighted')["recall"]
        f1 = metric4.compute(predictions=y_pred, references=y_test, average='weighted')["f1"]

        logger.info(f"Computed evaluation metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    except Exception as e:
        logger.error(f"Error in compute_metrics_eval: {e}")
        raise

def evaluate_model(model, tokenized_datasets, data_collator, device):
    """
    Evaluate the model on the test dataset and compute evaluation metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        tokenized_datasets (datasets.DatasetDict): The tokenized dataset.
        data_collator (transformers.DataCollator): The data collator for creating batches.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        dict: A dictionary with the computed metrics.
    """
    try:
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

        logger.info("Model evaluation completed")
        return compute_metrics_eval(y_pred, y_test)
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        raise

def generate_predictions(model, tokenized_datasets, device, id_to_category, num_samples=5):
    """
    Generate predictions for a specified number of samples from the test dataset.

    Args:
        model (torch.nn.Module): The model to use for generating predictions.
        tokenized_datasets (datasets.DatasetDict): The tokenized dataset.
        device (torch.device): The device to run the model on (CPU or GPU).
        id_to_category (dict): A dictionary mapping category IDs to category names.
        num_samples (int, optional): The number of samples to predict. Defaults to 5.

    Returns:
        pandas.DataFrame: A DataFrame containing the articles, true labels, predicted labels, true categories, and predicted categories.
    """
    try:
        test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=1)
        model.eval()

        predictions = []
        for i, batch in enumerate(test_dataloader):
            if i >= num_samples:
                break

            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}

            with torch.no_grad():
                outputs = model(**batch)

            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            predictions.append(predicted_label)

        data = {
            'article': [],
            'predicted_labels': [],
            'true_labels': [],
            'true_category': [],
            'predicted_category': []
        }
        for i in range(len(predictions)):
            data['article'].append(tokenized_datasets['test'][i]['article'])
            data['predicted_labels'].append(predictions[i])
            data['true_labels'].append(tokenized_datasets['test'][i]['labels'])
            data['true_category'].append(tokenized_datasets['test'][i]['category'])
            data['predicted_category'].append(id_to_category[predictions[i]])

        logger.info("Generated predictions for test dataset")
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error in generate_predictions: {e}")
        raise
