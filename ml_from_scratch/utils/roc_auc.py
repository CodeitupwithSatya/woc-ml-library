import numpy as np

def roc_curve(y_true, y_scores):
    """
    Compute ROC curve points from scratch

    Parameters:
    -----------
    y_true   : array-like, shape (m,)
               True binary labels (0 or 1)
    y_scores : array-like, shape (m,)
               Predicted probabilities

    Returns:
    --------
    fpr : array
    tpr : array
    thresholds : array
    """
    y_true = y_true.flatten()
    y_scores = y_scores.flatten()

    # Sort by decreasing score
    desc_order = np.argsort(-y_scores)
    y_true = y_true[desc_order]
    y_scores = y_scores[desc_order]

    thresholds = np.unique(y_scores)
    tpr = []
    fpr = []

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))

        tpr.append(TP / P if P != 0 else 0)
        fpr.append(FP / N if N != 0 else 0)

    return np.array(fpr), np.array(tpr), thresholds


def auc_score(fpr, tpr):
    """
    Compute AUC using trapezoidal rule
    """
    # Sort by FPR
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    auc = np.trapezoid(tpr, fpr)
    return auc
