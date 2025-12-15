import numpy as np
from .metrics import precision_recall_f1

def find_best_threshold(y_true, y_scores, metric="f1"):
    """
    Find optimal threshold for classification

    Parameters:
    -----------
    y_true   : array-like (m,)
    y_scores : predicted probabilities
    metric   : "f1", "precision", "recall", "accuracy"

    Returns:
    --------
    best_threshold : float
    best_score     : float
    scores         : dict (threshold -> metric value)
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    scores = {}

    best_threshold = 0.5
    best_score = -np.inf

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        prec, rec, f1, acc = precision_recall_f1(y_true, y_pred)

        if metric == "f1":
            score = f1
        elif metric == "precision":
            score = prec
        elif metric == "recall":
            score = rec
        elif metric == "accuracy":
            score = acc
        else:
            raise ValueError("Unsupported metric")

        scores[thresh] = score

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score, scores
