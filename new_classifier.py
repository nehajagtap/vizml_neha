import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix


def get_features(file_path):
    with open(file_path, 'rb') as handle:
        features = pickle.load(handle)
    return features

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=13),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]


train_features = get_features("/home/nehaj/infinity/altair/1.pkl") #path of train data(features)
train_ground_t = get_features("/home/nehaj/infinity/altair/1gt.pkl") #path of ground truth

test_features = get_features("/home/nehaj/infinity/altair/test_feat_new.pkl") #path of test data(features)
test_ground_t = get_features("/home/nehaj/infinity/altair/test_gt_new.pkl") #path of ground truth for test data

train_dataset = list(zip(train_features, train_ground_t))
test_dataset = list(zip(test_features, test_ground_t))

# train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

X_train, y_train = zip(*train_dataset)
X_test, y_test = zip(*test_dataset)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# X_train = np.nan_to_num(X_train, posinf=np.finfo('float32').max, neginf=-np.finfo('float32').max)
# X_test = np.nan_to_num(X_test, posinf=np.finfo('float32').max, neginf=-np.finfo('float32').max)
# X_train = np.clip(X_train, -np.finfo('float32').max, np.finfo('float32').max)
# X_test = np.clip(X_test, -np.finfo('float32').max, np.finfo('float32').max)

for name, clf in zip(names, classifiers):

        pipeline_clf = make_pipeline(StandardScaler(), clf)
        pipeline_clf.fit(X_train, y_train)
        y_pred = pipeline_clf.predict(X_test)
        score = pipeline_clf.score(X_test, y_test)
        print(f"{name}: {score}")

        # count number of ones and correct ones
        """ ones = np.sum(y_pred == 1)
        zeros = np.sum(y_pred == 0)
        correct_ones = np.sum(np.logical_and(y_pred == 1, y_test == 1))
        correct_zeros = np.sum(np.logical_and(y_pred == 0, y_test == 0))
        print(f"Ones: {ones}, Correct Ones: {correct_ones}")
        print(f"Zeros: {zeros}, Correct Zeros: {correct_zeros}")
        print("\n") """#

        report = classification_report(y_test, y_pred)
        print(report)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)

