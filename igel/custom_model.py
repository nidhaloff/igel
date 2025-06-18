class CustomModelBase:
    """
    Base class for custom model architectures in igel.
    Users can inherit from this class to define their own models.
    Required methods:
        - fit(X, y): Train the model on data X and labels y.
        - predict(X): Predict using the trained model on data X.
    """
    def fit(self, X, y):
        raise NotImplementedError("fit method must be implemented by the custom model.")

    def predict(self, X):
        raise NotImplementedError("predict method must be implemented by the custom model.") 