from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyperparameter_search(model,
                          method,
                          params,
                          x_train,
                          y_train,
                          **kwargs):
    search = GridSearchCV(model,
                          params,
                          **kwargs) if method == 'grid_search' else RandomizedSearchCV(model,
                                                                                       params,
                                                                                       **kwargs)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_score_, search.best_params_
