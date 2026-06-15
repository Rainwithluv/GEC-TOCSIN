"""
Evaluation metrics for LLM-generated text detection.
"""

import numpy as np
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score, accuracy_score, f1_score
)


def get_roc_metrics(real_preds, sample_preds):
    """
    Compute ROC metrics including AUC and optimal threshold.

    Args:
        real_preds: List of predictions for human-written texts
        sample_preds: List of predictions for LLM-generated texts

    Returns:
        Tuple of (roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy)
    """
    real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predicted_probs = real_preds + sample_preds

    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic for optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    predictions = [1 if prob >= optimal_threshold else 0 for prob in predicted_probs]
    conf_matrix = confusion_matrix(real_labels, predictions)
    precision = precision_score(real_labels, predictions)
    recall = recall_score(real_labels, predictions)
    f1 = f1_score(real_labels, predictions)
    accuracy = accuracy_score(real_labels, predictions)

    return (
        float(roc_auc), float(optimal_threshold), conf_matrix.tolist(),
        float(precision), float(recall), float(f1), float(accuracy)
    )


def get_metrics_with_threshold(real_preds, sample_preds, threshold):
    """
    Compute metrics with a fixed threshold.

    Args:
        real_preds: List of predictions for human-written texts
        sample_preds: List of predictions for LLM-generated texts
        threshold: Fixed threshold for classification

    Returns:
        Tuple of (roc_auc, threshold, conf_matrix, precision, recall, f1, accuracy)
    """
    real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predicted_probs = real_preds + sample_preds

    predictions = [1 if prob >= threshold else 0 for prob in predicted_probs]
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)

    conf_matrix = confusion_matrix(real_labels, predictions)
    precision = precision_score(real_labels, predictions)
    recall = recall_score(real_labels, predictions)
    f1 = f1_score(real_labels, predictions)
    accuracy = accuracy_score(real_labels, predictions)

    return (
        float(roc_auc), float(threshold), conf_matrix.tolist(),
        float(precision), float(recall), float(f1), float(accuracy)
    )
