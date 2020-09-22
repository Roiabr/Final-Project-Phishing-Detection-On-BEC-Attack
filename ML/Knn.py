from sklearn.neighbors import KNeighborsClassifier


def knn(CountTrainTf, CountTestTf, y_train, y_test, flag):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(CountTrainTf, y_train)
    knnScore = knn.score(CountTestTf, y_test)
    if flag is 0:
        print("Knn on header Score: ", knnScore)
    else:
        print("Knn on Body Score: ", knnScore)
    return knnScore, knn
