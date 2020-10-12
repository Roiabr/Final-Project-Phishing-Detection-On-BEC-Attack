from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def svm(CountTrainTf, CountTestTf, y_train, y_test, flag):
    Svm = SVC(kernel='rbf')
    Svm.fit(CountTrainTf, y_train)
    SvmScore = Svm.score(CountTestTf, y_test)
    Y_predict = Svm.predict(CountTestTf)
    if flag is 0:
        print("Svm on header Score: ", SvmScore)
        print(confusion_matrix(Y_predict, y_test))
    else:
        print("Svm on Body Score: ", SvmScore)
        print(confusion_matrix(Y_predict, y_test))
    return SvmScore, Svm






