from sklearn.tree import DecisionTreeClassifier


def decisionTreeClassifier(CountTrainTf, CountTestTf, y_train, y_test, flag):
    DT = DecisionTreeClassifier()
    DT.fit(CountTrainTf, y_train)
    DTScore = DT.score(CountTestTf, y_test)
    if flag is 0:
        print("Decision Tree Classifier on header Score: ", DTScore)
    else:
        print("Decision Tree Classifier on Body Score: ", DTScore)
    return DTScore, DT
