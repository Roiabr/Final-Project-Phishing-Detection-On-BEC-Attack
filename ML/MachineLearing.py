from ML import Random_Forest
import WordsToVector


class MachineLearning:
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train= y_train
        self.y_test= y_test
        self.train_vectors, self.test_vectors = WordsToVector.CountVector(X_train,X_test)




    def Ml(self,type):
        Random_Forest.random_forest(self.train_vectors, self.y_train,
                                    self.test_vectors, self.y_test)

