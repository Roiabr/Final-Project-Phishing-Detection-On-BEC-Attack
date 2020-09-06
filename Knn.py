from sklearn.neighbors import KNeighborsClassifier


def knn(trainX, trainY, testX, testY, numNeigh=3):
    # n_jobs means number of parallel jobs to run. -1 meansusing all processors
    model = KNeighborsClassifier(n_neighbors=numNeigh, n_jobs=-1)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY) * 100
    return acc
