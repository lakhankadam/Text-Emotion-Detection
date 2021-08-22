from sklearn.model_selection import train_test_split
from store_label import format_data
from sklearn.feature_extraction import DictVectorizer

vectorizer = DictVectorizer(sparse = True)

def get_split_data():
    X_all, y_all = format_data()
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test