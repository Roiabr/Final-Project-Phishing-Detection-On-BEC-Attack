from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def knn(CountTrainTf, CountTestTf, y_train, y_test, flag):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(CountTrainTf, y_train)
    knnScore = knn.score(CountTestTf, y_test)
    Y_predict = knn.predict(CountTestTf)
    if flag is 0:
        print("Knn on header Score: ", knnScore)
        print(confusion_matrix(Y_predict, y_test))
    else:
        print("Knn on Body Score: ", knnScore)
        print(confusion_matrix(Y_predict, y_test))
    return knnScore, knn
