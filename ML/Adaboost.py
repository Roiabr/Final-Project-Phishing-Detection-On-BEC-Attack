from sklearn.ensemble import AdaBoostClassifier


def adaBoost(X_train, y_train, X_test, y_test):
    boost = AdaBoostClassifier()
    # Train the model on training data
    boost.fit(X_train, y_train)
    acc = boost.score(X_test, y_test) * 100
    return acc
