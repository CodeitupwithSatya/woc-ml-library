import numpy as np

def learning_curve(model_class, X, y, train_sizes, **params):
    train_errors = []
    val_errors = []

    for size in train_sizes:
        X_train = X[:size]
        y_train = y[:size]

        model = model_class(**params)
        model.fit(X_train, y_train)

        train_errors.append(
            np.mean((model.predict(X_train) - y_train) ** 2)
        )
        val_errors.append(
            np.mean((model.predict(X) - y) ** 2)
        )

    return train_errors, val_errors
