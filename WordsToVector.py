from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords


def CountVector(X_train, X_test):
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)
    return train_vectors, test_vectors


def TfIdf(X_train, X_test):
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    train_vectors2 = tfidfconverter.fit_transform(X_train)
    test_vectors2 = tfidfconverter.transform(X_test)
    print(train_vectors2.shape, test_vectors2.shape)
    print(train_vectors2)
