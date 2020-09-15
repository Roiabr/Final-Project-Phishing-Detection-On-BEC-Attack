from sklearn.ensemble import RandomForestClassifier



def random_forest(X_train, y_train, X_test, y_test):
    # Instantiate model with 1000 decision trees
    forest_clf = RandomForestClassifier()
    # Train the model on training data
    forest_clf.fit(X_train, y_train)
    acc = forest_clf.score(X_test, y_test) * 100
    return acc, forest_clf
