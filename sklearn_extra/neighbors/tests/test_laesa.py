import numpy as np
from sklearn_extra.neighbors import LaesaClassifer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def test_laesa_classification():
    X, y = make_blobs(n_samples=30000, centers=2, n_features=2, cluster_std=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

    print("KNN algorithm launched")
    knn_model = KNeighborsClassifier(n_neighbors=1, algorithm="brute")
    knn_model.fit(X_train, y_train)
    # print("KNN pred:", X_test, knn_model.predict(X_test), y_test, sep="\n")
    print(knn_model.score(X_train, y_train))
    print(knn_model.score(X_test, y_test))

    print("LAESA algorithm launched")
    laesa_model = LaesaClassifer()
    laesa_model.fit(X_train, y_train)
    # print("Laesa pred:", X_test, laesa_model.predict(X_test), y_test, sep="\n")
    print(laesa_model.score(X_train, y_train))
    print(laesa_model.score(X_test, y_test))



if __name__ == "__main__":
    test_laesa_classification()