import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


def round_probabilities(probabilities, threshold):

    """
    Round probabilities to labels based on the given threshold

    Parameters
    ----------
    probabilities (np.ndarray of shape (n_samples)): Predicted probabilities
    threshold (float): Rounding threshold

    Returns
    -------
    labels (numpy.ndarray of shape (n_samples)): Labels
    """

    labels = np.zeros_like(probabilities, dtype=np.uint8)
    labels[probabilities >= threshold] = 1

    return labels


def specificity_score(y_true, y_pred):

    """
    Calculate specificity (true-negative rate) of predicted labels

    Parameters
    ----------
    y_true (numpy.ndarray of shape (n_samples)): Ground truth
    y_pred (numpy.ndarray of shape (n_samples)): Predicted labels

    Returns
    -------
    (float): Specificity score
    """

    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def classification_scores(y_true, y_pred, threshold=0.5):

    """
    Calculate binary classification metrics on predicted probabilities and labels

    Parameters
    ----------
    y_true (numpy.ndarray of shape (n_samples)): Ground truth
    y_pred (numpy.ndarray of shape (n_samples)): Predicted probabilities
    threshold (float): Rounding threshold

    Returns
    -------
    scores (dict): Dictionary of calculated scores
    """

    y_pred_labels = round_probabilities(y_pred, threshold=threshold)
    scores = {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'roc_auc': roc_auc_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred_labels),
        'recall': recall_score(y_true, y_pred_labels),
        'specificity': specificity_score(y_true, y_pred_labels),
        'f1': f1_score(y_true, y_pred_labels)
    }

    return scores
