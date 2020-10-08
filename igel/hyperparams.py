from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyperparameter_search(method, **kwargs):
    search = GridSearchCV(**kwargs) if method == 'grid_search' else RandomizedSearchCV(**kwargs)
    return search.best_estimator_, search.best_score_, search.best_params_
