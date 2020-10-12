from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def decisionTreeClassifier(CountTrainTf, CountTestTf, y_train, y_test, flag):
    DT = DecisionTreeClassifier()
    DT.fit(CountTrainTf, y_train)
    DTScore = DT.score(CountTestTf, y_test)
    Y_predict = DT.predict(CountTestTf)
    if flag is 0:
        print("Decision Tree Classifier on header Score: ", DTScore)
        print(confusion_matrix(Y_predict, y_test))
    else:
        print("Decision Tree Classifier on Body Score: ", DTScore)
        print(confusion_matrix(Y_predict, y_test))
    return DTScore, DT
