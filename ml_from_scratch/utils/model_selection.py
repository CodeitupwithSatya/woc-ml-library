import  numpy as np

def grid_search(model_class, param_grid, X, y, cv_func):
    best_score = -float("inf")
    best_params = None

    for params in param_grid:
        scores = cv_func(model_class, X, y, **params)
        score = sum(scores) / len(scores)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

def k_fold_cv(model_class, X, y, k=5, **model_params):
    """
    Generic K-Fold Cross Validation for regression models
    """
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    fold_size = m // k
    scores = []

    for i in range(k):
        val_start = i * fold_size
        val_end = min((i + 1) * fold_size, m)

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        scores.append(model.score(X_val, y_val))

    return scores

