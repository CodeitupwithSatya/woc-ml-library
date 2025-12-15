import numpy as np

def bias_variance_decomposition(model_class, X, y, test_idx,
                                n_repeats=10, sample_ratio=0.8, **model_params):
    """
    Bias-Variance decomposition using bootstrapping.
    """
    X_test = X[test_idx]
    y_test = y[test_idx].flatten()

    predictions = []

    for _ in range(n_repeats):
        sample_idx = np.random.choice(
            len(X), size=int(sample_ratio * len(X)), replace=True
        )

        X_sample = X[sample_idx]
        y_sample = y[sample_idx]

        model = model_class(**model_params)
        model.fit(X_sample, y_sample)

        pred = model.predict(X_test).flatten()
        predictions.append(pred)

    predictions = np.array(predictions)
    avg_pred = predictions.mean(axis=0)

    bias2 = np.mean((avg_pred - y_test) ** 2)
    variance = np.mean(np.var(predictions, axis=0))

    return bias2, variance
