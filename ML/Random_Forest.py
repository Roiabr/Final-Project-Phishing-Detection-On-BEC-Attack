from sklearn.ensemble import RandomForestClassifier


def random_forest(CountTrainTf, CountTestTf, y_train, y_test, flag):
    RF = RandomForestClassifier()
    RF.fit(CountTrainTf, y_train)
    RFScore = RF.score(CountTestTf, y_test)
    if flag is 0:
        print("Random Forest on header Score: ", RFScore)
    else:
        print("Random Forest on Body Score: ", RFScore)
    return RFScore, RF
