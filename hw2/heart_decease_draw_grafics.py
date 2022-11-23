import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from decision_tree import DTC
from knn import KNN

data_frame = pd.read_csv("heart.csv")
sns.countplot(data=data_frame, x="target", color="b")
plt.xlabel("Здоровье: 0 = здоров, 1 = болен")
plt.ylabel("Количество")
plt.title("Соотношение здоровые/больные => хорошее обучение")
plt.savefig(
    f"data_analysis/analyze_target_decease.png",
    transparent=False,
    facecolor="white",
    dpi=250,
)

pd.crosstab(data_frame.thal, data_frame.target).plot(kind="barh", color=["b", "g"])
plt.legend(["Здоровый", "Больной"])
plt.ylabel("Уровень заболевания")
plt.xlabel("Количество")
plt.title("Выше уровень => выше риск заболевания")
plt.savefig(
    f"data_analysis/analyze_level_decease.png",
    transparent=False,
    facecolor="white",
    dpi=250,
)


data_frame = pd.concat(
    [
        data_frame,
        pd.get_dummies(data_frame["cp"], prefix="cp"),
        pd.get_dummies(data_frame["thal"], prefix="thal"),
        pd.get_dummies(data_frame["slope"], prefix="slope"),
    ],
    axis=1,
)
data_frame.drop(columns=["cp", "thal", "slope"])

"""Масштабиование"""
Y = data_frame["target"]
X = data_frame.drop(columns=["target"])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=40
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


accuracies = {}

knn_searcher = GridSearchCV(
    estimator=KNN(),
    param_grid=[{"k": [range(1, 7)], "p_minkoswki": [range(1, 7)]}],
    cv=5,
)
knn_tuned = {"k": 1, "p_minkoswki": 2}

knn = KNN(k=1, p_minkowski=2)
knn.fit(X_train_scaled, Y_train)

accuracy = knn.score(X_test_scaled, Y_test)
accuracies["KNN"] = accuracy


leaf_size_grid = list(range(1, 50))
n_neighbors_grid = list(range(1, 30))
p_grid = [1, 2]

builtin_knn_searcher = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=[
        {"leaf_size": leaf_size_grid, "n_neighbors": n_neighbors_grid, "p": p_grid}
    ],
    cv=5,
)
builtin_knn_tuned = {"leaf_size": 1, "n_neighbors": 1, "p": 1}

builtin_knn = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
builtin_knn.fit(X_train_scaled, Y_train)

accuracy = builtin_knn.score(X_test_scaled, Y_test)
accuracies["Built-in KNN"] = accuracy


max_depth_grid = list(range(3, 40))
min_samples_split_grid = [5, 10, 20, 50, 100]

dtc_searcher = GridSearchCV(
    estimator=DTC(),
    param_grid=[
        {"max_depth": max_depth_grid, "min_samples_split": min_samples_split_grid}
    ],
    cv=5,
)
dtc_tuned = {"max_depth": 11, "min_samples_split": 5}

dtc = DTC(11, 5)
dtc.fit(X_train, Y_train)

accuracy = dtc.score(X_test, Y_test)
accuracies["Decision Tree"] = accuracy


max_depth_grid = list(range(3, 40))
min_samples_split_grid = [5, 10, 20, 50, 100]
criterion_grid = ["gini", "entropy"]

builtin_dtc_searcher = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=[
        {
            "max_depth": max_depth_grid,
            "min_samples_split": min_samples_split_grid,
            "criterion": criterion_grid,
        }
    ],
    cv=5,
)
builtin_dtc_tuned = {"criterion": "entropy", "max_depth": 29, "min_samples_split": 5}
if builtin_dtc_tuned is None:
    builtin_dtc_searcher.fit(X_train, Y_train)

builtin_dtc = sklearn.tree.DecisionTreeClassifier(
    criterion="entropy", max_depth=29, min_samples_split=5
)
builtin_dtc.fit(X_train, Y_train)

accuracy = builtin_dtc.score(X_test, Y_test)
accuracies["Built-in Decision Tree"] = accuracy


max_depth_grid = list(range(3, 40))
min_samples_split_grid = list(range(1, 7))

solver_grid = ["newton-cg", "lbfgs", "liblinear"]
penalty_grid = ["none", "l1", "l2", "elasticnet"]
C_grid = [100, 10, 1.0, 0.1, 0.01]

lr_searcher = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=[{"solver": solver_grid, "penalty": penalty_grid, "C": C_grid}],
    cv=5,
)
lr_tuned = {"C": 100, "penalty": "none", "solver": "newton-cg"}

lr = LogisticRegression(C=100, solver="newton-cg")
lr.fit(X_train_scaled, Y_train)

accuracy = lr.score(X_test_scaled, Y_test)
accuracies["Logistic Regression"] = accuracy


C_grid = [0.1, 1, 10, 100]
gamma_grid = [1, 0.1, 0.01, 0.001]
kernel_grid = ["rbf", "poly", "sigmoid"]

svc_searcher = GridSearchCV(
    estimator=SVC(),
    param_grid=[{"C": C_grid, "gamma": gamma_grid, "kernel": kernel_grid}],
    cv=5,
)
svc_tuned = {"C": 0.1, "gamma": 1, "kernel": "poly"}

svc = SVC(C=0.1, gamma=1, kernel="poly")
svc.fit(X_train_scaled, Y_train)

accuracy = svc.score(X_test_scaled, Y_test)
accuracies["SVC"] = accuracy


bayes_searcher = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=[{"var_smoothing": np.logspace(0, -9, num=100)}],
    cv=5,
)
bayes_searcher.fit(X_train_scaled, Y_train)

bayes = GaussianNB(var_smoothing=0.15)
bayes.fit(X_train_scaled, Y_train)

accuracy = bayes.score(X_test_scaled, Y_test)
accuracies["Naive Bayes"] = accuracy


with open("heart_decease_result.txt", "w") as file:
    for name, accuracy in accuracies.items():
        file.write(name + ": " + str(accuracy) + "\n")

y_head_knn = knn.predict(X_test_scaled)
y_head_builtin_knn = builtin_knn.predict(X_test_scaled)
y_head_dtc = dtc.predict(X_test)
y_head_builtin_dtc = builtin_dtc.predict(X_test)
y_head_lr = lr.predict(X_test_scaled)
y_head_svm = svc.predict(X_test_scaled)
y_head_bayes = bayes.predict(X_test_scaled)

cm_knn = confusion_matrix(Y_test, y_head_knn)
cm_b_knn = confusion_matrix(Y_test, y_head_builtin_knn)
cm_dtc = confusion_matrix(Y_test, y_head_dtc)
cm_b_dtc = confusion_matrix(Y_test, y_head_builtin_dtc)
cm_lr = confusion_matrix(Y_test, y_head_lr)
cm_svm = confusion_matrix(Y_test, y_head_svm)
cm_bayes = confusion_matrix(Y_test, y_head_bayes)

plt.figure(figsize=(24, 12))

plt.suptitle("Confusion Matrices", fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

cm = " Confusion Matrix"

plt.subplot(3,3,1)
plt.title("KNN" + cm)
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,2)
plt.title("Built-in KNN" + cm)
sns.heatmap(cm_b_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,3)
plt.title("Decision Tree" + cm)
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,4)
plt.title("Built-in Decision Tree" + cm)
sns.heatmap(cm_b_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,5)
plt.title("Logistic Regression" + cm)
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,6)
plt.title("SVM" + cm)
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,7)
plt.title("Naive Bayes" + cm)
sns.heatmap(cm_bayes,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.savefig(
    'heart_decease_result.png',
    transparent=False,
    facecolor='white',
    dpi=250,
)
