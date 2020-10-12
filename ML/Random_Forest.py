from sklearn.ensemble import RandomForestClassifier
# Confusion Matrix
from sklearn.metrics import confusion_matrix


def random_forest(CountTrainTf, CountTestTf, y_train, y_test, flag):
    RF = RandomForestClassifier()
    RF.fit(CountTrainTf, y_train)
    Y_predict = RF.predict(CountTestTf)
    RFScore = RF.score(CountTestTf, y_test)
    if flag is 0:
        print("Random Forest on header Score: ", RFScore)
        print(confusion_matrix(Y_predict, y_test))
    else:
        print("Random Forest on Body Score: ", RFScore)
        print(confusion_matrix(Y_predict, y_test))
    return RFScore, RF
