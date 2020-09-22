from sklearn.svm import SVC


def svm(CountTrainTf, CountTestTf, y_train, y_test, flag):
    Svm = SVC(kernel='rbf')
    Svm.fit(CountTrainTf, y_train)
    SvmScore = Svm.score(CountTestTf, y_test)
    if flag is 0:
        print("Svm on header Score: ", SvmScore)
    else:
        print("Svm on Body Score: ", SvmScore)
    return SvmScore, Svm






