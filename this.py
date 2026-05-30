from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

best_k = 1
best_accuracy = 0

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, knn.predict(X_test))
    print(f"  k={k:2d}  :  {accuracy * 100:.1f}%")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

f_model = KNeighborsClassifier(n_neighbors=best_k)
f_model.fit(X_train, y_train)

new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
             columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
y_final = f_model.predict(new_flower)
print(f"\nPredicted species: {y_final[0]}")