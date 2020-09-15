from sklearn.tree import DecisionTreeClassifier


def decisionTreeClassifier(X_train, y_train, X_test, y_test):
    DtreeClf = DecisionTreeClassifier()
    DtreeClf.fit(X_train, y_train)
    acc = DtreeClf.score(X_test, y_test) * 100
    return acc
