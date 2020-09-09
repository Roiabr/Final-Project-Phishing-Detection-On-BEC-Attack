from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from ML import DecisionTreeClassifier, Adaboost, Random_Forest
import pandas as pd
import WordsToVector

if __name__ == '__main__':
    FILENAME = "emails_dataset.csv"
    # fields = ['email_header', 'Body', 'label']
    # Extract_features.save_output(FILENAME, fields)
    # # read all emails,parser the email to a csv file
    # Extract_features.get_email_ham(FILENAME)
    # Extract_features.get_email_spam(FILENAME)

    # # make a dataframe of the header features and the label for the machine learning on header
    data = pd.read_csv(FILENAME)
    print(data.head())
    print(len(data))
    X = data['email_header']
    y = data['label']
#    print(len(data[data.label=='1']))
#    print(len(data[data.label=='0']))


    # We get the important word from the headers
    WordsToVector.getTheWordHeader(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    # Convert the header text to vector of number for machine learning
    # the convert the text to number using CountVector method
    train_vectors, test_vectors = WordsToVector.CountVector(X_train, X_test)

    #print(Svm.svmlinear(train_vectors, y_train, test_vectors, y_test))
    # print(Svm.svmpoly(train_vectors, y_train, test_vectors, y_test))
    # print(Svm.svmrbf(train_vectors, y_train, test_vectors, y_test))
    # print(Svm.svmsigmoid(train_vectors, y_train, test_vectors, y_test))
    print(Random_Forest.random_forest(train_vectors, y_train, test_vectors, y_test))


    # print(DecisionTreeClassifier.decisionTreeClassifier(train_vectors, y_train, test_vectors, y_test))
    # print(Adaboost.adaBoost(train_vectors, y_train, test_vectors, y_test))

    # model.fit(train_vectors, y_train)
    # print(model.score(test_vectors, y_test))
    #
    # y_pred = model.predict(test_vectors)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))

    # the convert the text to number using TfIdf method
    # train_vectors2, test_vectors2 = WordsToVector.TfIdf(X_train, X_test)

    # print(Svm.svmlinear(train_vectors, y_train, test_vectors, y_test))
    # print(Svm.svmpoly(train_vectors, y_train, test_vectors, y_test))
    # print(Svm.svmrbf(train_vectors, y_train, test_vectors, y_test))
    # print(Svm.svmsigmoid(train_vectors, y_train, test_vectors, y_test))
    # print(Random_Forest.random_forest(train_vectors, y_train, test_vectors, y_test))
    # print(DecisionTreeClassifier.decisionTreeClassifier(train_vectors, y_train, test_vectors, y_test))
    # print(Adaboost.adaBoost(train_vectors, y_train, test_vectors, y_test))
    #
    # model = RandomForestClassifier()
    #
    # model.fit(train_vectors2, y_train)
    # print(model.score(test_vectors2, y_test))
    # y_pred = model.predict(test_vectors)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))
    # check every email in the data and if spam move to new data for body classifcaion
    BodyDataFrame = pd.DataFrame(columns=['email_header', 'Body', 'label'])

    # for i in range(train_vectors.shape[0]):
    #     try:
    #         if model.predict(train_vectors[i]) == ['0']:
    #             df2 = pd.DataFrame([data.iloc[i]], columns=['Body', 'label'])
    #             print(df2)
    #     except:
    #         print("An exception occurred")
    #
