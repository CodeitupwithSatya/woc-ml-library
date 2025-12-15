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
