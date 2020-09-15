from sklearn.svm import SVC


def svmlinear(X_train, y_train, X_test, y_test):
    svclassifier1 = SVC(kernel='linear')
    svclassifier1.fit(X_train, y_train)
    acc = svclassifier1.score(X_test, y_test) * 100
    return acc


def svmpoly(X_train, y_train, X_test, y_test):
    svclassifier2 = SVC(kernel='poly')
    svclassifier2.fit(X_train, y_train)
    acc = svclassifier2.score(X_test, y_test) * 100
    return acc


def svmrbf(X_train, y_train, X_test, y_test):
    svclassifier3 = SVC(kernel='rbf')
    svclassifier3.fit(X_train, y_train)
    acc = svclassifier3.score(X_test, y_test) * 100
    return acc


def svmsigmoid(X_train, y_train, X_test, y_test):
    svclassifier4 = SVC(kernel='sigmoid')
    svclassifier4.fit(X_train, y_train)
    acc = svclassifier4.score(X_test, y_test) * 100
    return acc
