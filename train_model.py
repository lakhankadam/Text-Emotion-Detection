from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from split_data import get_split_data

def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc

def train_and_get_models():
    svc = SVC()
    lsvc = LinearSVC(random_state=123)
    rforest = RandomForestClassifier(random_state=123)
    dtree = DecisionTreeClassifier()

    clifs = [svc, lsvc, rforest, dtree]

    X_train, X_test, y_train, y_test = get_split_data()
    print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
    print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))
    for clf in clifs: 
        clf_name = clf.__class__.__name__
        train_acc, test_acc = train_test(clf, X_train, X_test, y_train, y_test)
        print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))
    return clifs