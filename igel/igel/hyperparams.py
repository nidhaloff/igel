from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyperparameter_search(model,
                          method,
                          params,
                          x_train,
                          y_train,
                          **kwargs):

    search = None
    if method == 'grid_search':
        search = GridSearchCV(model,
                              params,
                              **kwargs)

    elif method == 'random_search':
        search = RandomizedSearchCV(model,
                                    params,
                                    **kwargs)
    else:
        raise Exception("hyperparameter method must be grid_search or random_search")

    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_score_, search.best_params_
